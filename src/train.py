from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import hydra
import lightning as L
import rootutils
import torch
import os
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
import omegaconf
import collections
import typing
import torch
import functools
# Allow safe unpickling of functools.partial from trusted checkpoints (else newer Pytorch versions fail to load them)
torch.serialization.add_safe_globals([functools.partial,
                                      torch.optim.AdamW,torch.optim.lr_scheduler.CosineAnnealingLR, torch.optim.lr_scheduler.ReduceLROnPlateau,
                                      torch.optim.Adam, torch.optim.AdamW,torch.optim.lr_scheduler.CosineAnnealingLR, torch.optim.lr_scheduler.ReduceLROnPlateau,
                                      omegaconf.ListConfig, omegaconf.DictConfig, omegaconf.dictconfig.DictConfig,
                                      omegaconf.nodes.AnyNode, omegaconf.base.Metadata, omegaconf.base.ContainerMetadata,
                                      collections.defaultdict, typing.Any,
                                      list, dict, int])
# Set precision for float32 matrix multiplications to 'high' for better performance
torch.set_float32_matmul_precision('high')
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: omegaconf.DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # ============================================================
    # GPU VERIFICATION LOGGING
    # ============================================================
    log.info("=" * 80)
    log.info("GPU CONFIGURATION:")
    log.info(f"  • CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"  • GPU Count: {torch.cuda.device_count()}")
        log.info(f"  • Current Device: {torch.cuda.current_device()}")
        log.info(f"  • Device Name: {torch.cuda.get_device_name(0)}")
        log.info(f"  • CUDA Version: {torch.version.cuda}")
    log.info(f"  • Trainer Accelerator: {trainer.accelerator.__class__.__name__}")
    log.info(f"  • Trainer Strategy: {trainer.strategy.__class__.__name__}")
    log.info(f"  • Number of Devices: {trainer.num_devices}")
    if hasattr(cfg.trainer, 'precision'):
        log.info(f"  • Precision: {cfg.trainer.precision}")
    log.info("=" * 80)
    # ============================================================

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    # Optional testing phase
    test_metrics: Dict[str, Any] = {}
    if cfg.get("test"):
        log.info("Starting testing!")

        # Case 1: User supplied an explicit checkpoint
        if hasattr(cfg, "ckpt_path") and cfg.ckpt_path:
            ckpt_path = cfg.ckpt_path
            log.info(f"Using user-specified checkpoint for testing: {ckpt_path}")

        else:
            # Case 2: Use trainer's best checkpoint (only valid if training ran)
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if not ckpt_path:
                log.warning("No best checkpoint found! Using current model weights.")
                ckpt_path = None

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Final ckpt used: {ckpt_path}")

        test_metrics = trainer.callback_metrics

    # merge train and (optional) test metrics
    metric_dict = {**train_metrics, **test_metrics}

    # ============================================================
    # PROBE EVALUATION (if enabled)
    # ============================================================
    if cfg.get("eval_after_training", False):
        log.info("\n" + "="*80)
        log.info("STARTING AUTOMATIC PROBE EVALUATION")
        log.info("="*80)
        
        try:
            from src.eval_probes import evaluate_with_probes
            
            # IMPORTANT: Load best checkpoint (not last checkpoint) for evaluation
            # This ensures consistency with standalone eval_probes.py runs
            best_ckpt_path = None
            if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
                best_ckpt_path = trainer.checkpoint_callback.best_model_path
                if best_ckpt_path and Path(best_ckpt_path).exists():
                    log.info(f"Loading BEST checkpoint for probe evaluation: {best_ckpt_path}")
                    # Load the best checkpoint into the model
                    checkpoint = torch.load(best_ckpt_path)
                    model.load_state_dict(checkpoint['state_dict'])
                    log.info("✓ Best checkpoint loaded successfully")
                else:
                    log.warning("Best checkpoint path not found or doesn't exist. Using current model state (last checkpoint).")
            else:
                log.warning("No checkpoint callback found. Using current model state (last checkpoint).")
            
            # Save encoder state for debugging
            torch.save(model.encoder.state_dict(), "encoder_training.pth")
            log.info(f"Saved encoder state to encoder_training.pth for debugging")
            
            # ========================================
            # ENSURE CLEAN STATE FOR REPRODUCIBLE EMBEDDINGS
            # ========================================
            log.info("Clearing CUDA cache and ensuring clean state for probe evaluation...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Ensure model is in eval mode with no gradients
            model.eval()
            torch.set_grad_enabled(False)
            
            # Move model back to device to ensure clean device state
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            log.info(f"✓ Model in eval mode on {device} with clean state")
            
            # ========================================
            # DIAGNOSTIC SAVES - TRAINING PATH
            # ========================================
            log.info("\n" + "="*80)
            log.info("SAVING DIAGNOSTICS (TRAINING PATH)")
            log.info("="*80)
            
            import numpy as np
            debug_dir = Path("debug_comparison")
            debug_dir.mkdir(exist_ok=True)
            
            # 1. Save test dataloader samples
            log.info("Extracting first batch from test dataloader...")
            test_dl = datamodule.test_dataloader()
            test_batch = next(iter(test_dl))
            test_features, test_labels = test_batch
            
            np.savez(
                debug_dir / "train_path_test_batch.npz",
                features=test_features.cpu().numpy(),
                labels=test_labels.cpu().numpy()
            )
            log.info(f"Saved first test batch: {test_features.shape}, labels: {test_labels[:20]}")
            
            # 2. Save datamodule metadata
            datamodule_info = {
                'num_classes': datamodule.num_classes,
                'batch_size': datamodule.batch_size_per_device,
                'class_names': datamodule.classnames if hasattr(datamodule, 'classnames') else None,
                'teststream_shuffle': datamodule.teststream.shuffle if hasattr(datamodule, 'teststream') else None,
                'source': 'train.py',
                'datamodule_id': id(datamodule),
            }
            
            # Save test file order if available
            if hasattr(datamodule, 'teststream') and hasattr(datamodule.teststream, 'all_files'):
                test_files = datamodule.teststream.all_files
                datamodule_info['num_test_files'] = len(test_files)
                datamodule_info['first_5_files'] = [str(f) for f in test_files[:5]]
                datamodule_info['last_5_files'] = [str(f) for f in test_files[-5:]]
                log.info(f"Test dataset has {len(test_files)} files")
                log.info(f"First 5 files: {test_files[:5]}")
            
            import json
            with open(debug_dir / "train_path_datamodule_info.json", 'w') as f:
                json.dump(datamodule_info, f, indent=2, default=str)
            
            log.info(f"Saved datamodule info: {datamodule_info}")
            log.info("="*80 + "\n")
            
            probe_results = evaluate_with_probes(cfg, model, datamodule, call_source='train.py')
            
            # Merge probe results into metric dict
            metric_dict.update(probe_results)
            
            # Log probe results to loggers if available
            if logger:
                for lg in logger:
                    if hasattr(lg, "log_metrics"):
                        try:
                            lg.log_metrics(probe_results)
                        except Exception as e:
                            log.warning(f"Failed to log probe metrics to {lg}: {e}")
            
            log.info("Probe evaluation completed successfully!")
            
        except Exception as e:
            log.error(f"Probe evaluation failed: {e}")
            log.warning("Continuing without probe evaluation...")
    else:
        log.info("\nSkipping probe evaluation (eval_after_training=False)")

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: omegaconf.DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    os.chdir(cfg.paths.root_dir)
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()

# Archive

Code kept for provenance but no longer part of the pipeline that produces the paper's
results. Nothing here is imported by `src/` or `scripts/`; the reference analysis that
selected these files checked every `.py`, `.sh`, `.sub` and `.yaml` in the repository.

- `legacy_launchers/` — shell launchers from earlier campaigns. All of them pass
  `experiment=` values whose config no longer exists (`aug_supcon_15class`,
  `anomaly_qcd_vs_higgs_raw`, `vanillasupcon_6class_pretrain`, …), so they cannot run
  as they stand.
- `legacy_condor/` — per-model `.sub` files written by hand before the submitters in
  `scripts/new_exp/` generated them. Superseded: those scripts write the `.sub` and
  submit it in one step, which is what the paper's runs used.
- `oneoff_analysis/` — exploratory plotting and diagnostics that no longer feed any
  table or figure.
- `superseded/` — modules replaced by the current pipeline. `gmm_explain.py` predates
  `scripts/xai/`, which splits the same analysis into the six auditable steps.

- `configs/` — Hydra configs, one subdirectory per group, mirroring `configs/`.
  Selected by the same reference analysis, cross-checked against what is actually on
  EOS: a config was only archived if no published run came from it.
  - `experiment/` — the pre-`new_exp` SupCon and VICReg trainings. Their `new_exp/`
    replacements produced every SupCon and VICReg number in the paper. Note that the
    sibling `vcreg_*` and `fm_testing_*` configs were **kept**, because they are the
    ones that produced the published VCReg and CE encoders.
  - `experiment_new_exp/` — `new_exp/vcreg_dmodel{32,64,128,256}` and its five
    pilots. A different VCReg recipe (lr 1e-3, 40 epochs, checkpoint on
    `val/vcreg_loss`) that was trained, measured and set aside; its encoders are in
    `anomaly_pipeline/new_exp/archive_vcreg_20260728/`. Kept out of `configs/` because
    the name invites the assumption that it is the published VCReg, and it is not.
  - `model_scheduler/` — the whole `model/scheduler` config group. Never selected by
    anything: every config declares its scheduler inline under `model.scheduler`.
  - `model/`, `data/`, `paths/`, `logger/`, `hparams_search/` — an unused MLP, two
    pre-v2 dataset definitions, path files for machines this work never ran on
    (4090, A6000, CINECA), the SaaS loggers of the upstream template (`mlflow`, `csv`
    and `tensorboard` are kept), and the Optuna searches of the earlier campaign.

- `superseded_submitters/` — `submit_training_vcreg.py`, the launcher for the abandoned
  VCReg recipe. It targets `new_exp/vcreg_dmodel{d}` (now in `configs/experiment_new_exp/`)
  over ten seeds — `[7, 42, 100, 137, 1000, 10000, 12345, 31337, 100000, 999999]` — none
  of which exist on EOS. The published VCReg is submitted by
  `scripts/new_exp/submit_training_new_exp.py --models vcreg` with seeds 0–4.

- `configs/experiment_ablation_arch/` — the 44-config `*_arch_*` architecture sweep
  (dff, dropout, heads, layers, per model), plus its launcher in
  `superseded_submitters/submit_training_ablation_backbone.py` and its two `paths`
  entries. A second attempt at axes that already exist and ran: `ablation/dff` has 64
  runs and `ablation/dropout`, `nheads`, `nlayers` 58–60 each, while this family's own
  target `ablation/backbone` holds two runs from 2026-06-27, three minutes apart, and
  no results. 28 of the 44 did not even compose, and none was ever committed.

  Their `*_arch_base` mixins declare `- ablation/training/backbone_X` while sitting in
  that same group, so Hydra resolved it to
  `experiment/ablation/training/ablation/training/backbone_X`. Changing those four
  lines to the bare `- backbone_X` is all it would take to revive the family, if the
  architecture axis is ever wanted at a separate log root.

- `ablation/` — the whole appendix-ablation apparatus: 172 configs
  (`configs_experiment/{root,loss,training}/`), 35 submitters and wrappers (`scripts/`)
  and 11 `paths` entries (`configs_paths/`). Retired because no current stage runs them;
  the results they produced remain on EOS under `anomaly_pipeline/ablation/<axis>/`.
  Their submitters still write the `stream_output`/`stream_error` lines the CERN schedd
  rejects, so reviving an axis means removing those two lines first.

  TWO PIECES WERE DELIBERATELY NOT ARCHIVED, because production depends on them:
  - the five backbone mixins, moved to `configs/experiment/backbone/`. They are
    inherited by 22 live configs (SupCon, SimCLR, VICReg and the VCReg probes), so
    archiving them would have broken the paper's training. Their `paths` entry moved
    with them as `configs/paths/backbone_cern.yaml`, keeping its `log_dir` unchanged.
  - `wrapper_train.sh`, moved to `scripts/new_exp/`, which is the executable
    `submit_training_new_exp.py` submits.

  The move was verified by resolving all 75 live experiment configs before and after:
  75 identical, 0 changed, 0 broken.

To restore something, move it back and check its `experiment=` target still exists.

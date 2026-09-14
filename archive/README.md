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

  Also `raw_ae_qcd_new_exp.sub` and `raw_ae_allsm_new_exp.sub`, the two raw-feature AE
  submitters from which no published row came. The first runs the tree without the
  SM-only normalisation (`anomaly_qcd_vs_higgs_raw_nosparse_cern`) and writes into
  `ad_results/raw`, the directory of the published baseline. Its wrapper skips a seed
  whose summary already exists, so running both kept whichever ran first, with nothing
  in the output to say which. The published row comes from `raw_ae_qcd_smnorm_new_exp.sub`,
  as the `raw_experiment` field of all five `strategies_summary.json` confirms. The
  second trains the AE on all twelve SM classes instead of QCD alone. Both still carry
  the `stream_output` lines the CERN schedd rejects.

- `superseded_submitters/xai/` — interpretability submitters from which no published
  figure or table came. The paper's chain is written out step by step in stage 4 of the
  top-level README.
  - `xai_full.sub` + `wrapper_xai_full.sh` run steps 01–06 end to end with the
    mixture chosen by `01_select_k.py` (unprojected, K from ARI). That is a different
    mixture from the paper's K = 7 on PCA 64. The only paper output that ever came
    through it is `matched_sm_hh4b.npz`, which has since been rebuilt by
    `04_profile_and_rank.py --save-matched`.
  - `xai_k7.sub` + `wrapper_xai_k7.sh` run steps 03, 05 and 06 at K = 7, but step 03
    for HH→4b scores the whole test set rather than the matched array, writes to
    `03_assign` instead of `03_assign_matched`, and step 06 reads its `profile_meta`
    from `rank_k7/`, which predates the b-tag fix.
  - `rank_k7.sub` + `wrapper_rank_k7.sh` write `rank_k7/`, the pre-fix ranking; the
    paper's is `rank_k7_sm/`.
  - `select_k_v3_pca.sub` scans a PCA at 99% variance (209 dimensions). The paper's
    mixtures come from `select_k_v4_pca_aggressive.sub` at 64 dimensions.

  They still point at `scripts/xai/submit/wrapper_*.sh`, so they cannot run from here.

  `newsig_vicreg.sub` queued the four VICReg jobs of the HH->bbtautau evaluation in a
  separate file, from a resubmission; `wrapper_newsig.sh` handles VICReg like any other
  model, seed set included, so those four rows now sit in `newsig.sub` and the stage
  mirrors the CASE one: one submitter for the four encoders, one for the raw baseline.

  Archived with them, the earlier K-selection machinery, which no published figure or
  table came from either. `01_select_k.py` (BIC/ARI scan) and `02_fit_gmm.py` fit the
  mixture the way the first drafts did; the paper's mixture is fitted by
  `select_k_interpretable.py` and the value of K comes from `select_k_profiles.py`, both
  of which stay in `scripts/xai/`. With them go the three workflows that were their only
  callers: `submit_select_k.py` + `wrapper_select_k.sh` (step 01 on a d128 seed-0 run),
  `submit_xai_pipeline.py` + `wrapper_xai_pipeline.sh` (step 02), and the ARI-scaling
  diagnostic `diagnose_ari_scaling.py` with `k_diagnostic.sub`, `ari_scaling.sub` and
  their two wrappers, which measured how ARI moves with sample size.

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

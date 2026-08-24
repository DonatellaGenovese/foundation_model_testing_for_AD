<div align="center">

# Explaining Anomalies in Collider Data via Learned Latent Representations

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## About

Code for the paper *Explaining Anomalies in Collider Data via Learned Latent
Representations*. Transformer encoders are trained on 12 Standard-Model process
classes from COLLIDE-2V under four objectives (SupCon, SimCLR, VCReg, VICReg); the
embeddings are frozen and a small autoencoder is trained on them, on QCD only, to flag
signal processes as anomalies. A Gaussian mixture fitted on the SM latent space is
then used to interpret *why* an event was flagged, by comparing flagged events with
their local SM neighbourhood in physics space.

Built on the data-loading and Lightning-Hydra scaffolding of
[pploner/foundation_model_testing](https://github.com/pploner/foundation_model_testing),
itself based on the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

Retired code lives in [`archive/`](archive/README.md) with a note on why each group
was set aside.

## Installation

```bash
git clone https://github.com/dgenoves/foundation_model_testing_for_AD.git
cd foundation_model_testing_for_AD
apptainer build fm_testing.sif fm_testing.def
```

Behaviour is controlled through the Hydra configs in `configs/`; set defaults there
and override only what changes in a `configs/experiment/*.yaml`.

Every submitter below accepts `--dry-run`, which prints what would be submitted
without touching the queue. **Use it first** — a full sweep is 20 GPU jobs.

---

## 1. Vectorisation and preprocessing

Both stages come from
[pploner/foundation_model_testing](https://github.com/pploner/foundation_model_testing),
which this repository is built on. Transforms and normalisers are unchanged from it.

Vectorisation reads the source `.parquet`, keeps the top-k objects per group (12 jets,
8 electrons/muons/photons), zero-pads to a fixed length and appends each group's true
multiplicity as a trailing scalar. Preprocessing then transforms and normalises the
result. Output goes to `<label>/vectorized/` and `<label>/preprocessed/`; **a different
set of options needs a different `label`**, or the existing tree is reused unchanged.

### Preprocessing for pretraining and probe evaluation

Preprocessing with filter the events with all features not reconstructed:

```bash
python scripts/submit_vectorization_jobs.py experiment=fm_testing_12class_nosparse_dmodel256_cern
python scripts/submit_preprocessing_jobs.py experiment=fm_testing_12class_nosparse_dmodel256_cern

```

Preprocessing without filter:

```bash
python scripts/submit_vectorization_jobs.py experiment=fm_testing_12class_nosparse_dmodel256_cern \
    data.drop_empty_events=false data.label=v2_sparse_12class_highlevel
python scripts/submit_preprocessing_jobs.py experiment=fm_testing_12class_nosparse_dmodel256_cern \
    data.label=v2_sparse_12class_highlevel
```

The filter drops events in which every jet, electron, muon and photon slot has `PT == 0`.

This stage fits the normalisation statistics median and IQR per feature group and writes them to `<label>/preprocessed/norm_stats.json`. Everything downstream reuses
that file.

### Proxy signals and CASE signals

These are normalised with the statistics fitted above, never their own: a signal that
helped set the scale would be measured against a scale it had already shifted.

```bash
python scripts/preprocess_smnorm.py        # 12 SM + 3 Higgs proxies
python scripts/prepare_newsig_smnorm.py    # QCD + 5 further proxies
python scripts/prepare_case_smnorm.py      # QCD + 7 CASE signals

# without the empty-event filter (writes to a `sparse` label instead)
python scripts/prepare_newsig_smnorm.py --keep-empty
python scripts/prepare_case_smnorm.py --keep-empty
```

Each copies the SM-only `norm_stats.json` into its output directory, runs the
preprocessing with `mode: apply_only` and `force: true`, then re-reads the file and
exits non-zero if `num_examples_fit` changed — the only way `apply_only` could fail
silently. `force` is needed because the stage otherwise skips when statistics are
already present; here they are the input, not the result.

`preprocess_smnorm.py` has no `--keep-empty`: it reuses an already vectorised tree and
only applies the statistics.

To do the same by hand for a new sample:

```bash
cp <sm_label>/preprocessed/norm_stats.json <new_label>/preprocessed/
python scripts/submit_preprocessing_jobs.py experiment=<experiment> \
    preprocess.mode=apply_only preprocess.force=true
```

> **TODO — the signal datasets must be rebuilt.** `newsig` and `case` currently hold
> every process that was tried, not the ones that end up in the paper. Rebuild both
> with the final selection before release, so the published datasets match the
> published tables.

If the source production changes, rescan the event map first —
`has_enough_events` raises on a file it cannot look up:

```bash
python src/utils/nEvents_scan/scan_parquet_nevent.py
```

---

## 2. Training

Four encoders plus the supervised baseline, each over `d_model ∈ {32, 64, 128, 256}`
and 5 seeds.

```bash
# everything the paper reports
python scripts/new_exp/submit_training_new_exp.py \
    --models supcon simclr vcreg vicreg ce

# one model, one dimension
python scripts/new_exp/submit_training_new_exp.py --models vcreg --dmodels 256
```

**The five rows do not share a seed set or a config namespace.** They were trained in
two campaigns and the submitter now carries the right pairing per model, so the
command above reproduces the published encoders — but do not pass a global `--seeds`
expecting it to.

| `--models` | paper row | config | seeds |
|---|---|---|---|
| `supcon` | SupCon | `new_exp/supcon_dmodel{d}` | 7, 42, 137, 1337, 31337 |
| `simclr` | SimCLR | `new_exp/simclr_dmodel{d}` | 7, 42, 137, 1337, 31337 |
| `vicreg` | VICReg | `new_exp/vicreg_dmodel{d}` | 7, 42, 12345, 1337, 31337 |
| `vcreg` | VCReg | `vcreg_12class_nosparse_dmodel{d}_cern` | 0, 1, 2, 3, 4 |
| `ce` | CE baseline | `fm_testing_12class_nosparse_dmodel{d}_cern` | 0, 1, 2, 3, 4 |

VCReg and CE come from the earlier campaign, which is why their configs sit outside
`new_exp/`. This is verifiable rather than a naming accident: their published
checkpoints under `new_exp/<run_dir>/seed_{0..4}` are byte-identical to
`anomaly_pipeline/encoder_seeds/`, and their `config_tree.log` records lr 1.5e-4,
wd 5e-4, 50 epochs and checkpointing on `val/acc` — the settings of the older config,
not of `new_exp/vcreg_dmodel{d}`. That newer VCReg recipe was trained and set aside;
its encoders are in `new_exp/archive_vcreg_20260728/` and its configs in
`archive/configs/experiment_new_exp/`.

VCReg d256 is the encoder the interpretability stage runs on, so this is the pairing
that matters most.

The remaining registry entries are the VICReg augmentation arms, kept for provenance.


The raw-feature baselines need no training — the autoencoder and the linear probe are
fitted directly on the preprocessed kinematics in stage 3.

A single run without the submitter:

```bash
python src/train.py experiment=fm_testing_12class_nosparse_dmodel256_cern seed=7
```

---

## 3. Probes and anomaly detection

### Linear probes

```bash
python scripts/new_exp/submit_eval_probes_new_exp.py \
    --models supcon simclr vcreg vicreg

condor_submit scripts/new_exp/raw_linear_probe_new_exp.sub    # raw-feature baseline
```

One linear layer on the frozen embeddings; per-class and macro AUROC, aggregated over
seeds into `aggregated_summary.json`. CE is not probed — being a classifier, its
per-class AUROC comes from its own test predictions.

> The probe is **not** a reliable proxy for anomaly detection. Two independent
> measurements here move the two metrics in opposite directions — SimCLR's temperature
> and VICReg's `d_model` — so a configuration selected by probe AUROC can be worse at
> the task the paper reports.

### Anomaly detection, proxy signals

The autoencoder is trained on QCD only and scored by reconstruction MSE. The operating
threshold is calibrated on validation and stored in the checkpoint, so every later
evaluation transfers it rather than recomputing it.

```bash
# embedding-based, all models — VBF H->bb, HH->4b, ggH->tautau
python scripts/new_exp/submit_ad_new_exp.py --smnorm \
    --models supcon simclr vcreg vicreg

# raw-feature baseline (strategy mse_qcd is the published row)
condor_submit scripts/new_exp/raw_ae_qcd_smnorm_new_exp.sub

# four further proxies — HH->bbtautau, VVV, VH, tttt
condor_submit scripts/xai/submit/newsig.sub
```

### Anomaly detection, CASE signals

Inference only: nothing is trained and no threshold is recalibrated, so these
processes are held out in the strongest sense.

```bash
condor_submit scripts/xai/submit/case_ad.sub     # the four encoders
condor_submit scripts/xai/submit/case_raw.sub    # raw-feature baseline
```

Check the measured false-positive rate on QCD in the output: it confirms the
transferred threshold still lands where it should — 0.096 ± 0.004 against a nominal
0.10 on the CASE production, 0.090 ± 0.002 for the raw baseline.


(TODO: sistemare i segnali che usiamo nel paper)

---

## 4. Interpretability

Six steps in `scripts/xai/`. The autoencoder flags; the mixture only interprets, and
is never used as an anomaly score.

| Step | Script | Output |
|---|---|---|
| 1 | `01_select_k.py` | BIC/ARI scan |
| 2 | `02_fit_gmm.py` | `gmm_K{k}.pkl` |
| 3 | `03_assign_flagged.py` | flag rate per component (QCD / non-QCD SM / signal) |
| 4 | `04_profile_and_rank.py` | physics profile per component, Wasserstein ranking |
| 5 | `05_robustness_kpm2.py` | stability at K ± 2 |
| 6 | `06_ae_mechanism.py` | per-dimension AE residual vs. the ranked observable |

```bash
condor_submit scripts/xai/submit/xai_full.sub     # HH->4b, end to end
condor_submit scripts/xai/submit/xai_case.sub     # a CASE signal, step 4 only
```

Two choices govern this stage, and neither follows the criterion stated in earlier
drafts.

**The mixture is fitted on a 64-dimensional PCA of the embedding, not on all 256.**
The autoencoder keeps scoring the full embedding, so no anomaly score changes; only
the partition is projected. In the unprojected space no K between 3 and 12 gives a
partition whose components are all populated — even at K=3 one component holds 4% of
the uniform share — whereas at 64 dimensions occupancy is monotone in K.

**K = 7 comes from the criterion in `select_k_profiles.py`**: the finest partition
whose components are all populated and none of which are duplicates, the duplicate
test calibrated by permutation rather than against a fixed distance. BIC decreases
monotonically over the whole range in every space measured and never selects; ARI
never reaches 0.8 in the unprojected space, so the threshold quoted in earlier drafts
is unattainable there.

`select_k_interpretable.py` runs the earlier ARI/BIC/profile-distance scan, kept
because the PCA comparison in the appendix comes from it.

---

## 5. Ablations

The appendix ablations — augmentation strategy, loss hyperparameters, `d_model` — were
run on a reduced subset of the 12-class dataset. They are separate runs from the
production ones and the numbers are not interchangeable: different subset, and a
different backbone (`n_heads=8`, `n_layers=6`).

Their configs and submitters now live in [`archive/ablation/`](archive/README.md), since
they no longer run as part of any current stage. The results they produced are on EOS
under `anomaly_pipeline/ablation/<axis>/`. To re-run one, move the axis back into
`configs/experiment/` and `scripts/`, and note that its submitter still carries the
`stream_output` lines the CERN schedd now rejects (see stage 1).

What did **not** move is the five shared backbone definitions the ablation happened to
introduce, now in `configs/experiment/backbone/`: the production SupCon, SimCLR, VICReg
and probe configs inherit them, so they are part of the live tree despite the name.

---

## Layout

```
configs/experiment/new_exp/   production configs, one per model and d_model
scripts/new_exp/              submitters for training, probes, anomaly detection
scripts/xai/                  interpretability pipeline (6 steps) and its submitters
configs/experiment/backbone/  shared backbone mixins (inherited by production configs)
src/                          Lightning modules, data and preprocessing
archive/                      retired code, see archive/README.md
```

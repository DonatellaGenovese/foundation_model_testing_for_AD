<div align="center">

# Explaining Anomalies in Collider Data via Learned Latent Representations

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## About

Code for the paper *Diagnosing Learned Event Representations for Anomaly Detection in High-Luminosity LHC Searches*. Transformer encoders are trained on 12 Standard-Model process classes from COLLIDE-2V under four objectives (SupCon, SimCLR, VCReg, VICReg); the embeddings are frozen and a small autoencoder is trained on them, on QCD only, to flag signal processes as anomalies. A Gaussian mixture fitted on the SM latent space is then used to interpret *why* an event was flagged, by comparing flagged events with their local SM neighbourhood in physics space.

Built on the data-loading and Lightning-Hydra scaffolding of
[pploner/foundation_model_testing](https://github.com/pploner/foundation_model_testing),
itself based on the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

Retired code lives in [`archive/`](archive/README.md) with a note on why each group
was set aside.

## Installation

```bash
git clone https://github.com/DonatellaGenovese/foundation_model_testing_for_AD.git
cd foundation_model_testing_for_AD
apptainer build fm_testing.sif fm_testing.def
```

Behaviour is controlled through the Hydra configs in `configs/`; set defaults there
and override only what changes in a `configs/experiment/*.yaml`.

Every submitter below accepts `--dry-run`, which prints what would be submitted
without touching the queue. **Use it first** (a full sweep is 20 GPU jobs.)

---

## 1. Vectorisation and preprocessing

Both stages come from
[pploner/foundation_model_testing](https://github.com/pploner/foundation_model_testing),
which this repository is built on. Transforms and normalisers are unchanged from it.

Vectorisation reads the source `.parquet`, keeps the top-k objects per group (12 jets,
8 electrons/muons/photons), zero-pads to a fixed length and appends each group's true
multiplicity as a trailing scalar. Preprocessing then transforms and normalises the result. Output goes to `<label>/vectorized/` and `<label>/preprocessed/`; **a different set of options needs a different `label`**, or the existing tree is reused unchanged.

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

### Held-out signals

The held-out processes are normalised with the statistics fitted above.

```bash
python scripts/preprocess_smnorm.py        # 12 SM + VBF H->bb, HH->4b, ggH->tautau
python scripts/prepare_newsig_smnorm.py    # QCD + HH->bbtautau
python scripts/prepare_case_smnorm.py      # QCD + H->aa->4b, H->aa->4tau, Z'->n(mumu)
```


To do the same by hand for a new sample:

```bash
cp <sm_label>/preprocessed/norm_stats.json <new_label>/preprocessed/
python scripts/submit_preprocessing_jobs.py experiment=<experiment> \
    preprocess.mode=apply_only preprocess.force=true
```

If the source production changes, rescan the event map first:
`has_enough_events` raises on a file it cannot look up:

```bash
python src/utils/nEvents_scan/scan_parquet_nevent.py
```

### Or download the data instead

Stage 1 takes hours on the batch system, and its output is published: the vectorised and
preprocessed 12-class trees, which are what training and the probes read.

```bash
bash scripts/download_dataset.sh                 # both trees, about 44 GB
bash scripts/download_dataset.sh preprocessed    # only the preprocessed tree, 18 GB
bash scripts/download_dataset.sh vectorized      # only the vectorised tree, 26 GB
```

They unpack into `data/v2_12class_nosparse_highlevel/`, and a run reads them by
overriding one path:

```bash
python src/train.py experiment=fm_testing_12class_nosparse_dmodel256_cern seed=7 \
    paths.eos_data_dir=$PWD/data
```

Both trees are published because the data loader walks the vectorised tree before every
run — it skips the shards that already exist — so with the preprocessed tree alone it
would try to read the source production, which lives in a CERN project space
(`paths.dataset_dir`).

The held-out signal trees above are not published in full. The test splits the
interpretability stage needs come with its own package; see stage 4.

---

## 2. Training

Four encoders plus the supervised baseline, each over `d_model ∈ {32, 64, 128, 256}`
and 5 seeds.

```bash
# all the training together 
python scripts/new_exp/submit_training_new_exp.py \
    --models supcon simclr vcreg vicreg ce

# to select one model and dimension 
python scripts/new_exp/submit_training_new_exp.py --models vcreg --dmodels 256
```

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
per-class AUROC comes from its own test predictions, which training writes to
`seed_*/test_predictions/test_logits_and_labels.npz`:

```bash
python scripts/aggregate_ce_auroc.py    # CE rows of Tables 4 and 8
```

**Probing a published encoder.** The encoders the paper reports are published, so a
probe can be run without training anything:

```bash
bash scripts/download_weights.sh encoders     # 2.24 GB into data/weights/encoders/
```

The file name carries the run — `<model>_d<dim>_seed<seed>.ckpt` — and the epoch the run
stopped at is inside the checkpoint, together with the architecture. One probe is then
one command:

```bash
python src/eval_probes.py experiment=vcreg_12class_nosparse_dmodel256_cern \
    ckpt_path=data/weights/encoders/vcreg_d256_seed3.ckpt seed=3 \
    eval.linear_probe.max_epochs=50 \
    paths.eos_data_dir=$PWD/data paths.output_dir=$PWD/outputs/probe_vcreg_d256_seed3
```

`experiment=` selects the probe protocol — split, labels, dataset — not the architecture,
which comes from the checkpoint. It is `new_exp/supcon_dmodel{dim}`,
`new_exp/simclr_dmodel{dim}`, `new_exp/vicreg_dmodel{dim}` for the other three, and the
one above for VCReg. Averaging the five seeds of a model is what
`scripts/aggregate_probe_results.py` does, and its output is the
`aggregated_summary.json` the tables read.

The seed sets differ per model: VCReg 0–4, SupCon and SimCLR 7/42/137/1337/31337, VICReg
7/42/12345/1337/31337. Note also that the runs recorded in `probe_results/` and
`ad_results/` name two directories that were later renamed: `selfsupcon` is SimCLR and
`vicreg_stable` is VICReg.

### Anomaly detection, proxy signals

The autoencoder is trained on QCD only and scored by reconstruction MSE. The operating
threshold is calibrated on validation and stored in the checkpoint.

```bash
# embedding-based, all models — VBF H->bb, HH->4b, ggH->tautau
python scripts/new_exp/submit_ad_new_exp.py --smnorm \
    --models supcon simclr vcreg vicreg

# raw-feature baseline (strategy mse_qcd is the published row)
condor_submit scripts/new_exp/raw_ae_qcd_smnorm_new_exp.sub

# further proxy — HH->bbtautau; build its dataset first, or the jobs race to build it
python scripts/prepare_newsig_smnorm.py
condor_submit scripts/xai/submit/newsig.sub          # the four encoders
condor_submit scripts/xai/submit/newsig_raw.sub      # raw-feature baseline
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

### Scoring with published weights


```bash
bash scripts/download_weights.sh     # encoders + autoencoders, 2.3 GB into data/weights/

python scripts/infer_new_signals.py --dataset case --model vcreg --dmodel 256 --seed 3 \
    --encoder-ckpt data/weights/encoders/vcreg_d256_seed3.ckpt \
    --ae-ckpt data/weights/autoencoders/vcreg_d256_seed3_ae.ckpt \
    --out-dir outputs/ad_infer
```

`--dataset` picks the held-out set:

| value | processes scored against QCD |
|---|---|
| `proxy` | VBF H→bb, HH→4b, ggH→ττ |
| `newsig` | HH→bbττ |
| `case` | `hToAA_4b_ma60` = H→aa→4b with m_a = 60 GeV, `hToAA_4tau_ma15` = H→aa→4τ with m_a = 15 GeV, `HVdilep_Zp1000_piD2_mumu` = hidden-valley Z′ (1 TeV) → n(μμ) |

---

## 4. Interpretability

Reproduces Section 4.3 (Fig. 2, Table 6, the localisation, top-observable and Spearman
figures) with the VCReg encoder (`d_model = 256`, seed 3) and the K = 7 mixture fitted on
a 64-dimensional PCA of the SM training embeddings.

### From scratch (CERN batch system and EOS)

```bash
source scripts/xai/paths.sh     # input and output paths; export any of them to override

# 1. embeddings of the 12 SM classes and the signals
condor_submit scripts/xai/submit/extract_xai_emb.sub

# 2. SM + HH->4b event array
python scripts/xai/04_profile_and_rank.py --ckpt-path $ENC --signal-label 13 \
    --vectorized-dir $FMD/v2_nosparse_higgs_allsm_highlevel/vectorized/test \
    --preproc-split-dir $FMD/v2_nosparse_higgs_smnorm_highlevel/preprocessed/test \
    --save-matched $MH --output-dir $(dirname $MH)

# 3. mixtures, K = 5..12, in PCA 64 and unprojected
condor_submit scripts/xai/submit/select_k_v4_pca_aggressive.sub
condor_submit scripts/xai/submit/select_k_v3.sub

# 4. choice of K (Appendix A.4)
condor_submit scripts/xai/submit/select_k_profiles.sub
python paper/figures/make_k_occupancy.py

# 5. SM + Z'->n(mumu) event array
python scripts/xai/build_matched_case.py --case-label HVdilep_Zp1000_piD2_mumu \
    --signal-label 20 --sm-matched $MH --ckpt $ENC --output $MV

# 6. assignments, Wasserstein ranking, K +- 2 check and Spearman, for both signals
bash scripts/xai/run_steps_03_06.sh

# 7. figures and Table 6
bash scripts/xai/make_figures.sh
```

Re-running steps 1 and 3 does not give back the published embeddings and mixture
exactly: extraction is not seeded, and the mixture is reproduced only inside
`fm_testing.sif`. To reproduce the paper's numbers, use the published ones below.

### With the public data (≈15 min, CPU)

```bash
bash scripts/download_xai_data.sh     # 4 GB into data/: embeddings, mixture, checkpoints, test sets
jupyter nbconvert --to notebook --execute notebooks/xai_reproduce_load.ipynb
```

The notebook builds the event arrays, runs steps 03, 04 and 06, draws the figures and
Table 6 into `outputs/xai_reproduce/figures/`, and checks every number against the paper.
It needs a Python environment with the repository's requirements and Jupyter:
`fm_testing.sif` does not include Jupyter.

---

## License

MIT, see [`LICENSE`](LICENSE). The scaffolding this repository is built on
([pploner/foundation_model_testing](https://github.com/pploner/foundation_model_testing),
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)) carries
its own licence.

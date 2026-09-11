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
condor_submit scripts/xai/submit/newsig.sub          # VCReg, SupCon, SimCLR
condor_submit scripts/xai/submit/newsig_vicreg.sub   # VICReg
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

---

## 4. Interpretability

The autoencoder flags; the mixture only interprets, and is never used as an anomaly
score. The whole stage runs on one encoder — VCReg, `d_model = 256`, seed 3 and one
mixture: K = 7, diagonal covariance, fitted on a 64-dimensional PCA of the SM training
embeddings. The autoencoder is the stage-3 one and keeps scoring the full 256-dim
embedding; only the partition is projected.

The two signals interpreted are HH→4b (label 13) and Z′→n(μμ) (label 20, from the CASE production of stage 1). Run the steps in this order; the paths are those of the
paper's runs, so change `NE`, `XP` and `FMD` to your own.

[`notebooks/xai_reproduce_load.ipynb`](notebooks/xai_reproduce_load.ipynb) loads the saved
embeddings (step 1) and K = 7 mixture (step 3), runs steps 2 and 5–7 into a separate
output tree, then checks every number against the value printed in the paper. Its
inputs (about 4 GB) are shared from CERNBox: `bash scripts/download_xai_data.sh`
downloads, verifies and unpacks them into `data/`, where the notebook looks for them.

```bash
NE=/eos/user/d/dgenoves/anomaly_pipeline/new_exp
XP=/eos/user/d/dgenoves/anomaly_pipeline/xai_paper
FMD=/eos/user/d/dgenoves/foundation_model_testing_data
RUN=vcreg_12class_nosparse_dmodel256_cern
EMB=$NE/xai_embeddings_smnorm/$RUN/encoder_seed_3/embeddings
ENC=$NE/$RUN/seed_3/checkpoints/epoch_014.ckpt
AE=$NE/ad_results/$RUN/encoder_seed_3/mse_normal/checkpoints/ae-epochepoch=49.ckpt
GMM=$XP/k_selection_v3/vcreg_d256_seed3_diag_pca64/gmm_K7.pkl
MH=$XP/vcreg_d256_seed3_smnorm/04_profile/matched_sm_hh4b.npz
MV=$XP/case_HVdilep_Zp1000_piD2_mumu_d256_seed3/matched_sm_HVdilep_Zp1000_piD2_mumu.npz
PCA="--pca-dim 64 --pca-embeddings-dir $EMB --pca-seed 3"
```

**1. Embeddings of the 12 SM classes and the signals.** The AD runs only embed QCD and the signals, so the mixture needs its own extraction (only `DMODEL=256` is used):

```bash
condor_submit scripts/xai/submit/extract_xai_emb.sub
```

**2. Matched array** — SM and HH→4b test events with their embedding and physics
observables, the population every later step profiles:

```bash
python scripts/xai/04_profile_and_rank.py --ckpt-path $ENC --signal-label 13 \
    --vectorized-dir $FMD/v2_nosparse_higgs_allsm_highlevel/vectorized/test \
    --preproc-split-dir $FMD/v2_nosparse_higgs_smnorm_highlevel/preprocessed/test \
    --save-matched $MH --output-dir $(dirname $MH)
```

Without `--gmm-path` the script stops once the array is saved: the array does not
depend on the mixture, and step 3 needs it to fit one.

**3. Mixtures.** `select_k_interpretable.py` fits K = 5…12 in each space:

```bash
condor_submit scripts/xai/submit/select_k_v4_pca_aggressive.sub  # PCA 64 -> gmm_K7.pkl (16, 32 unused)
condor_submit scripts/xai/submit/select_k_v3.sub                 # unprojected, for the comparison (256 only)
```

**4. Choice of K** (Appendix A.4). `select_k_profiles.py` reuses the mixtures above,
fits K = 3, 4 itself, and records the occupancy of each component:

```bash
condor_submit scripts/xai/submit/select_k_profiles.sub   # unprojected and PCA 64
python paper/figures/make_k_occupancy.py                 # k_occupancy.pdf
```

A component is populated if it holds at least `max(200, 0.2 N_SM / K)` events. In PCA
64 every component is populated up to K = 7 and not beyond, so K = 7 is the finest
usable partition; in the unprojected space no K in 3–12 qualifies. The script also
tests for duplicate components, but no pair is flagged at any K, so occupancy alone
decides.

**5. Matched array for Z′→n(μμ)** — the CASE signal events next to the same SM block:

```bash
python scripts/xai/build_matched_case.py --case-label HVdilep_Zp1000_piD2_mumu \
    --signal-label 20 --sm-matched $MH --ckpt $ENC --output $MV
```

**6. Steps 03–06, per signal:**

| Step | Script | Output | Used for |
|---|---|---|---|
| 03 | `03_assign_flagged.py` | `k7_<sig>_pca64_d256_seed3/03_assign_matched/` | localisation, top-observable, the Fig. 2 assignments |
| 04 | `04_profile_and_rank.py` | `rank_k7_sm/<sig>/` | Table 6, per-component distributions |
| 05 | `05_robustness_kpm2.py` | `k7_<sig>_pca64_d256_seed3/05_robustness/` | the K ± 2 check of Appendix A.4 |
| 06 | `06_ae_mechanism.py` | `k7_<sig>_pca64_d256_seed3/06_ae_mechanism/` | Spearman figures |

```bash
for spec in "hh4b 13 $MH" "hvdilep 20 $MV"; do
  read t S M <<< "$spec"
  K7=$XP/k7_${t}_pca64_d256_seed3
  python scripts/xai/03_assign_flagged.py --matched-npz $M --signal-label $S \
      --gmm-path $GMM --ae-checkpoint $AE --output-dir $K7/03_assign_matched \
      --fpr 0.10 --ylim 0.95 $PCA
  python scripts/xai/04_profile_and_rank.py --matched-npz $M --signal-label $S \
      --gmm-path $GMM --ae-checkpoint $AE --output-dir $XP/rank_k7_sm/$t \
      --min-frac 0.05 --fpr 0.10 $PCA
  python scripts/xai/05_robustness_kpm2.py --embeddings-dir $EMB --matched-npz $M \
      --signal-label $S --ae-checkpoint $AE --k 7 --gmm-dir $(dirname $GMM) \
      --output-dir $K7/05_robustness --fpr 0.10 --min-frac 0.05 $PCA
  python scripts/xai/06_ae_mechanism.py --matched-npz $M --gmm-path $GMM \
      --ae-checkpoint $AE --profile-meta $XP/rank_k7_sm/$t/profile_meta.json \
      --output-dir $K7/06_ae_mechanism $PCA
done
```

**7. Figures and tables of the paper:**

```bash
R=$XP/rank_k7_sm; D=paper/figures/xai
H=$XP/k7_hh4b_pca64_d256_seed3; V=$XP/k7_hvdilep_pca64_d256_seed3

# Fig. 2, component profiles (the script reads <run-dir>/04_profile/)
mkdir -p $R/hh4b/04_profile
cp $R/hh4b/profile_meta.json $R/hh4b/04_profile/
ln -sf $MH $R/hh4b/04_profile/matched_sm_hh4b.npz
python scripts/xai/plot_04_profiles.py --run-dir $R/hh4b \
    --assignments $H/03_assign_matched/assignments.npz \
    --mark '$HH \to 4b$:4,5' --mark '$Z^{\prime} \to n(\mu\mu)$:2' \
    --output $D/component_profiles_K7.pdf

# Spearman (only C2 is shown for Z')
python scripts/xai/plot_06_convergence.py --run-dir $H --output $D/spearman_hh4b_K7.pdf
python scripts/xai/plot_06_convergence.py --run-dir $V --components 2 \
    --output $D/spearman_hvdilep_K7.pdf

# localisation, top observable, Table 6
cp $H/03_assign_matched/plots/flagged_assignment.pdf $D/localisation_hh4b_K7.pdf
cp $V/03_assign_matched/plots/flagged_assignment.pdf $D/localisation_hvdilep_K7.pdf
python paper/figures/make_top_observable.py    # top_observable_K7.pdf
python paper/make_wasserstein_table.py         # sections/xai/wasserstein_side_by_side.tex

# supplementary: per-component distributions
cp $R/hh4b/plots/hh4b_vs_sm_k5.pdf              $D/dist8_hh4b_C5_K7.pdf
cp $R/hh4b/plots/hh4b_vs_sm_k4.pdf              $D/dist8_hh4b_C4_K7.pdf
cp $R/hvdilep/plots/hv_zp1000_mumu_vs_sm_k2.pdf $D/dist8_hvdilep_C2_K7.pdf
cp $R/hvdilep/plots/hv_zp1000_mumu_vs_sm_k5.pdf $D/dist8_hvdilep_C5_K7.pdf
cp $R/hh4b/plots/physics_per_component.pdf      $D/physics_per_component_K7.pdf
```

---

---

## License

MIT, see [`LICENSE`](LICENSE). The scaffolding this repository is built on
([pploner/foundation_model_testing](https://github.com/pploner/foundation_model_testing),
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)) carries
its own licence.

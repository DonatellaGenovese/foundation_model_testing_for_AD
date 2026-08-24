"""Optional PCA projection for the GMM stage of the XAI pipeline.

WHY THIS IS SEPARATE FROM THE AE. The autoencoder — and therefore the anomaly
score, the calibrated thresholds and every AD number in the paper — always sees the
full embedding. Only the mixture that partitions the SM into regions is projected.
Reducing the dimension here changes the resolution of the *interpretation*, never
the detector, which is what makes the projection a post-hoc analysis choice rather
than a modification of the model under study.

Concretely, in every step that uses both: score the AE on Z, assign the GMM on
project(Z). Never project the array handed to the AE.

WHY A PROJECTION AT ALL. In the unprojected 256-dimensional space the mixture is
poorly conditioned: independent fits agree only at ARI 0.60-0.72, occupancy is
non-monotonic in K (full at K=5, 9, 11 but not at 6, 7, 8, 10, 12), and no K meets
a stability threshold. At 64 dimensions, retaining 52.7% of the variance, ARI rises
to 0.84 and occupancy becomes monotone, so the resolution limit is well defined.
The 0.99-variance run settles which effect is responsible: it keeps 209 of 256
dimensions, so it is nearly pure rotation with negligible truncation, and it makes
the ARI *worse* — the gain therefore comes from the truncation, not from the change
of basis. (A diagonal covariance is not rotation invariant, so the projection does
change the model class, not only the dimension.)

The PCA is refitted here rather than pickled. It is fully determined by the SM
train embeddings, the component count and the seed, so refitting reproduces the
exact basis the stored mixtures were fitted in — but the caller must pass the same
`pca_dim` and `seed` used then, or the mixture would be applied in a basis it never
saw. That is the one way to get silently wrong assignments here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from .constants import SM_INDICES
from .io_embeddings import filter_classes, load_train_val_test


def build_sm_pca(
    embeddings_dir: Path,
    pca_dim: float,
    seed: int = 3,
    verbose: bool = True,
) -> Optional[PCA]:
    """Fit PCA on the SM training embeddings, exactly as the K scans did.

    `pca_dim` follows the same convention as select_k_interpretable.py and
    select_k_profiles.py: 0 disables the projection, a value of 1 or more is an
    explicit component count, a value below 1 a fraction of the variance to retain.
    sklearn switches on the argument's *type*, so an integral float such as 64.0
    must be cast or it would be read as a variance target and rejected.

    Fitted on SM train alone: validation and the matched arrays are only ever
    transformed by it, never used to define it.
    """
    if not pca_dim or pca_dim <= 0:
        return None

    splits = load_train_val_test(Path(embeddings_dir))
    X_tr, _ = filter_classes(*splits["train"], SM_INDICES)
    n_comp = int(pca_dim) if pca_dim >= 1 else pca_dim
    pca = PCA(n_components=n_comp, svd_solver="full", whiten=False, random_state=seed)
    pca.fit(X_tr)
    if verbose:
        print(f"PCA for the GMM stage: {X_tr.shape[1]} -> {pca.n_components_} dims, "
              f"{pca.explained_variance_ratio_.sum():.4f} of the variance retained "
              f"(fitted on {len(X_tr):,} SM train events)")
    return pca


def project(pca: Optional[PCA], Z: np.ndarray) -> np.ndarray:
    """Transform for the GMM, or pass through when no projection is configured."""
    return Z if pca is None else pca.transform(Z)


def check_gmm_dims(gmm, Z_gmm: np.ndarray) -> None:
    """Fail loudly when the mixture and the array disagree on dimensionality.

    A mismatch means the mixture was fitted in a different space than the one it is
    being applied to — the failure mode this module's docstring warns about. It
    would otherwise surface as a shape error deep inside sklearn, or worse, not at
    all when two spaces happen to share a dimension.
    """
    expected = int(gmm.means_.shape[1])
    got = int(Z_gmm.shape[1])
    if expected != got:
        raise ValueError(
            f"GMM was fitted on {expected} dimensions but is being applied to "
            f"{got}. Pass the same --pca-dim (and seed) used to fit it: a mixture "
            f"from a PCA space cannot be applied to raw embeddings, or vice versa."
        )

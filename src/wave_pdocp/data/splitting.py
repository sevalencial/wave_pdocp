import numpy as np


def split_indices(
    n_samples: int, fractions=(0.75, 0.15, 0.10), seed=42, strategy="random"
) -> dict[str, np.ndarray]:
    """Largest-remainder counts; for small datasets reserve one row per split.

    The strategy argument is explicit for future blocked/extrapolation splits.
    Only random interpolation splitting is implemented in this milestone.
    """
    if strategy != "random":
        raise NotImplementedError(f"Split no implementado: {strategy}")
    f = np.asarray(fractions, dtype=float)
    if (
        f.shape != (3,)
        or not np.all(np.isfinite(f))
        or np.any(f <= 0)
        or not np.isclose(f.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Las fracciones deben ser tres valores positivos que sumen 1")
    if n_samples < 3:
        raise ValueError("Se necesitan al menos tres muestras PASS para dividir")
    raw = n_samples * f
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts), kind="stable")[: n_samples - counts.sum()]:
        counts[i] += 1
    for i in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        counts[donor] -= 1
        counts[i] = 1
    shuffled = np.random.default_rng(seed).permutation(n_samples)
    parts = np.split(shuffled, np.cumsum(counts)[:-1])
    return dict(
        zip(("train_indices", "validation_indices", "test_indices"), parts, strict=True)
    )

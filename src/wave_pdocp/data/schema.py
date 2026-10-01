import numpy as np

FIELD_NAMES = (
    "wave_speed",
    "wave_speed_half",
    "damping",
    "control_mask",
    "initial_displacement",
    "initial_velocity",
    "target_displacement",
    "target_velocity",
)
METRIC_COLUMNS = (
    "absolute_residual",
    "relative_residual",
    "optimality_error",
    "optimality_relative_error",
    "terminal_error",
    "relative_terminal_error",
    "max_displacement_error",
    "max_velocity_error",
    "control_energy",
    "optimal_cost",
    "zero_control_cost",
    "cost_reduction",
    "gramian_min_eigenvalue",
    "gramian_max_eigenvalue",
    "gramian_condition_number",
    "system_condition_number",
    "gramian_weighted_symmetry_error",
    "gramian_energy_coordinate_symmetry_error",
    "state_spectral_abscissa",
    "state_spectral_radius",
    "adjoint_norm",
)


def validate_dataset_arrays(samples: dict, metadata: dict, splits: dict) -> None:
    n, p = metadata["grid_size"], metadata["parameter_dimension"]
    m = len(samples["sample_id"])
    if samples["sample_id"].shape != (m,):
        raise ValueError("sample_id debe ser un vector")
    if metadata.get("split_created") and not splits:
        raise ValueError("La metadata declara splits que no están almacenados")
    if (
        not np.issubdtype(samples["sample_id"].dtype, np.integer)
        or np.unique(samples["sample_id"]).size != m
    ):
        raise ValueError("sample_id debe contener enteros únicos")
    shapes = {
        name: (m, n + 1 if name == "wave_speed_half" else n) for name in FIELD_NAMES
    }
    shapes.update(mu=(m, p), final_adjoint=(m, 2 * n), final_state=(m, 2 * n))
    for name, shape in shapes.items():
        if samples[name].shape != shape or not np.all(np.isfinite(samples[name])):
            raise ValueError(f"Shape o valores no finitos en {name}")
    if samples["status"].shape != (m,) or not set(samples["status"]).issubset(
        {"PASS", "WARNING", "FAIL"}
    ):
        raise ValueError("Estados inválidos")
    if metadata.get("role") == "training_dataset" and np.any(
        samples["status"] != "PASS"
    ):
        raise ValueError("El dataset automático solo admite muestras PASS")
    if splits:
        expected_names = {"train_indices", "validation_indices", "test_indices"}
        if set(splits) != expected_names:
            raise ValueError("Faltan splits")
        for indices in splits.values():
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("Los índices de split deben ser vectores de enteros")
        merged = np.concatenate(list(splits.values()))
        if not np.array_equal(np.sort(merged), np.arange(m)):
            raise ValueError(
                "Los splits deben ser disjuntos y cubrir todas las muestras"
            )

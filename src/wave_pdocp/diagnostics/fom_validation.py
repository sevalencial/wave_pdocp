from dataclasses import asdict, dataclass, field

import numpy as np

from ..fom.reconstruction import reconstruct_final_state
from ..fom.solution import FOMSolution
from .fom import compute_fom_diagnostics


@dataclass(frozen=True)
class ValidationConfig:
    safe_epsilon: float = 1e-15
    relative_residual_tolerance: float = 1e-9
    optimality_tolerance: float = 1e-9
    weighted_symmetry_tolerance: float = 1e-9
    positivity_tolerance: float = 1e-10
    cost_tolerance: float = 1e-10
    spectral_tolerance: float = 1e-9
    gramian_condition_warning: float = 1e10
    system_condition_warning: float = 1e8
    near_zero_eigenvalue_warning: float = 1e-12
    control_energy_warning: float = 1e6
    relative_terminal_error_warning: float = 0.1
    runtime_warning_seconds: float = 10.0
    sensitivity_outlier_factor: float = 10.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Las tolerancias deben ser positivas y finitas")


@dataclass
class ValidationResult:
    status: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def classify_metrics(
    metrics: dict, config: ValidationConfig, numerical_warnings=()
) -> ValidationResult:
    m, c = metrics, config
    failures, warnings = [], list(numerical_warnings)
    # A singular Gramian can have infinite condition number without invalidating K.
    if any(
        not np.isfinite(v) for k, v in m.items() if k != "gramian_condition_number"
    ) or np.isnan(m["gramian_condition_number"]):
        failures.append("nonfinite_diagnostic")
    for metric, tolerance in (
        ("relative_residual", c.relative_residual_tolerance),
        ("optimality_relative_error", c.optimality_tolerance),
        ("gramian_weighted_symmetry_error", c.weighted_symmetry_tolerance),
    ):
        if m[metric] >= tolerance:
            failures.append(metric)
    spectral_scale = max(1.0, abs(m["gramian_max_eigenvalue"]))
    if m["gramian_min_eigenvalue"] < -c.positivity_tolerance * spectral_scale:
        failures.append("gramian_not_positive")
    elif m["gramian_min_eigenvalue"] <= c.near_zero_eigenvalue_warning * spectral_scale:
        warnings.append("gramian_near_singular")
    cost_scale = max(1.0, abs(m["zero_control_cost"]))
    if m["optimal_cost"] > m["zero_control_cost"] + c.cost_tolerance * cost_scale:
        failures.append("optimal_cost_exceeds_zero_control")
    if min(m["control_energy"], m["optimal_cost"]) < -c.cost_tolerance * cost_scale:
        failures.append("negative_energy_or_cost")
    if m["state_spectral_abscissa"] > c.spectral_tolerance * max(
        1.0, m["state_spectral_radius"]
    ):
        failures.append("unstable_dissipative_family")
    for metric, limit in (
        ("gramian_condition_number", c.gramian_condition_warning),
        ("system_condition_number", c.system_condition_warning),
        ("control_energy", c.control_energy_warning),
        ("relative_terminal_error", c.relative_terminal_error_warning),
    ):
        if m[metric] > limit:
            warnings.append(metric)
    return ValidationResult(
        "FAIL" if failures else "WARNING" if warnings else "PASS",
        failures + warnings,
        m,
    )


def validate_fom(
    solution: FOMSolution, config: ValidationConfig, numerical_warnings=()
) -> ValidationResult:
    arrays = [
        getattr(solution, name)
        for name in (
            "state_operator",
            "control_operator",
            "state_adjoint",
            "control_adjoint",
            "gramian",
            "semigroup_T",
            "rhs",
            "final_adjoint",
            "residual",
        )
    ]
    arrays.extend(asdict(solution.fields).values())
    if not all(np.all(np.isfinite(a)) for a in arrays):
        return ValidationResult("FAIL", ["nonfinite_array", *numerical_warnings])
    if not np.all(np.isfinite(reconstruct_final_state(solution))):
        return ValidationResult("FAIL", ["nonfinite_final_state", *numerical_warnings])
    d = compute_fom_diagnostics(solution, config.safe_epsilon)
    metrics = {
        "absolute_residual": d.weighted_residual_norm,
        "relative_residual": d.relative_residual_norm,
        "optimality_error": d.terminal_optimality_error_norm,
        "optimality_relative_error": d.terminal_optimality_relative_error,
        "terminal_error": d.terminal_error_norm,
        "relative_terminal_error": d.relative_terminal_error,
        "max_displacement_error": d.displacement_max_error,
        "max_velocity_error": d.velocity_max_error,
        "control_energy": d.control_energy,
        "optimal_cost": d.objective_value,
        "zero_control_cost": d.uncontrolled_objective_value,
        "cost_reduction": d.cost_reduction,
        "gramian_min_eigenvalue": d.gramian_min_eigenvalue,
        "gramian_max_eigenvalue": d.gramian_max_eigenvalue,
        "gramian_condition_number": d.gramian_condition_number,
        "system_condition_number": d.system_condition_number,
        "gramian_weighted_symmetry_error": d.gramian_weighted_symmetry_error,
        "gramian_energy_coordinate_symmetry_error": d.gramian_relative_asymmetry,
        "state_spectral_abscissa": d.state_max_real_eigenvalue,
        "state_spectral_radius": d.state_spectral_radius,
        "adjoint_norm": d.adjoint_norm,
    }
    result = classify_metrics(metrics, config, numerical_warnings)
    if np.any(solution.fields.damping < 0):
        result.status = "FAIL"
        result.reasons.append("negative_damping_outside_campaign")
    return result

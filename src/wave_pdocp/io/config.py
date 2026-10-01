from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..diagnostics.fom_validation import ValidationConfig


class ConfigSection(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_default=True
    )


class ExperimentConfig(ConfigSection):
    name: str = "wave_speed_amplitude_sweep"
    seed: int = Field(default=42, ge=0)


class DomainConfig(ConfigSection):
    length: float = Field(default=1.0, gt=0)
    final_time: float = Field(default=2.0, gt=0)


class GridConfig(ConfigSection):
    n_interior: int = Field(default=100, ge=1)


class ControlConfig(ConfigSection):
    beta: float = Field(default=100.0, gt=0)


class ParameterConfig(ConfigSection):
    dimension: Literal[1] = 1
    min: float = 0.0
    max: float = 0.3
    n_samples: int = Field(default=100, ge=2)
    sampling: Literal["uniform_grid"] = "uniform_grid"
    parameterization: Literal["wave_speed_sine_amplitude_v1"] = (
        "wave_speed_sine_amplitude_v1"
    )

    @model_validator(mode="after")
    def valid_interval(self):
        if not -1 < self.min < self.max < 1:
            raise ValueError("La familia requiere -1 < min < max < 1")
        return self


class MeshConfig(ConfigSection):
    enabled: bool = True
    sizes: list[int] = Field(default_factory=lambda: [50, 100, 200])
    representative_samples: int = Field(default=8, ge=1)
    decrease_factor_tolerance: float = Field(default=1.05, ge=1)
    absolute_error_floor: float = Field(default=1e-8, gt=0)

    @model_validator(mode="after")
    def valid_sizes(self):
        if (
            len(self.sizes) < 3
            or self.sizes != sorted(set(self.sizes))
            or min(self.sizes) < 1
        ):
            raise ValueError(
                "Refinamiento requiere al menos tres tamaños positivos crecientes"
            )
        return self


class DatasetConfig(ConfigSection):
    format: Literal["hdf5"] = "hdf5"
    version: str = "1.0"
    split_strategy: Literal["random"] = "random"
    fractions: tuple[float, float, float] = (0.75, 0.15, 0.10)

    @model_validator(mode="after")
    def valid_fractions(self):
        if min(self.fractions) <= 0 or abs(sum(self.fractions) - 1) > 1e-12:
            raise ValueError("Las tres fracciones deben ser positivas y sumar 1")
        return self


class OutputConfig(ConfigSection):
    dataset: str = "data/datasets/wave_pdocp_fom.h5"
    snapshots: str = "outputs/parameter_sweep/validated_snapshots.h5"
    metrics: str = "outputs/metrics/fom_parameter_sweep.csv"
    refinement: str = "outputs/metrics/fom_mesh_refinement.csv"
    summary: str = "outputs/parameter_sweep/summary.json"
    figures: str = "outputs/parameter_sweep/figures"


class SweepConfig(ConfigSection):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    domain: DomainConfig = Field(default_factory=DomainConfig)
    discretization: GridConfig = Field(default_factory=GridConfig)
    optimal_control: ControlConfig = Field(default_factory=ControlConfig)
    parameter_space: ParameterConfig = Field(default_factory=ParameterConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    mesh_refinement: MeshConfig = Field(default_factory=MeshConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)


def load_sweep_config(path: Path) -> SweepConfig:
    with path.open(encoding="utf-8") as stream:
        return SweepConfig.model_validate(yaml.safe_load(stream))

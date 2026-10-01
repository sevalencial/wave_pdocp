import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from ..data.schema import validate_dataset_arrays
from ..data.splitting import split_indices


def write_dataset(
    path: Path,
    samples: dict,
    metadata: dict,
    splits: dict | None = None,
    overwrite=False,
) -> None:
    splits = splits or {}
    validate_dataset_arrays(samples, metadata, splits)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Ya existe {path}; use --overwrite explícitamente")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as stream:
            group = stream.create_group("metadata")
            group.attrs["json"] = json.dumps(metadata, sort_keys=True, allow_nan=False)
            for key in (
                "grid_size",
                "domain_length",
                "final_time",
                "beta",
                "parameterization_name",
                "dataset_version",
            ):
                group.attrs[key] = metadata[key]
            group = stream.create_group("samples")
            for name, value in samples.items():
                if np.asarray(value).dtype.kind in "UO":
                    group.create_dataset(
                        name,
                        data=np.asarray(value, dtype=object),
                        dtype=h5py.string_dtype("utf-8"),
                    )
                else:
                    group.create_dataset(name, data=value, compression="gzip")
            group = stream.create_group("splits")
            for name, indices in splits.items():
                group.create_dataset(name, data=indices)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_dataset(path: Path) -> tuple[dict, dict, dict]:
    with h5py.File(path, "r") as stream:
        metadata = json.loads(stream["metadata"].attrs["json"])
        samples = {
            name: value.asstr()[:] if h5py.check_string_dtype(value.dtype) else value[:]
            for name, value in stream["samples"].items()
        }
        splits = {name: value[:] for name, value in stream["splits"].items()}
    validate_dataset_arrays(samples, metadata, splits)
    return samples, metadata, splits


def build_validated_dataset(
    archive_path: Path, output_path: Path, overwrite=False
) -> dict:
    samples, metadata, _ = read_dataset(archive_path)
    keep = samples["status"] == "PASS"
    accepted = {name: values[keep] for name, values in samples.items()}
    n = int(np.sum(keep))
    config = metadata["config"]
    splits = (
        split_indices(
            n,
            config["dataset"]["fractions"],
            config["experiment"]["seed"],
            config["dataset"]["split_strategy"],
        )
        if n >= 3
        else {}
    )
    metadata = {
        **metadata,
        "role": "training_dataset",
        "source_archive": str(archive_path),
        "accepted_samples": n,
        "split_created": bool(splits),
    }
    write_dataset(output_path, accepted, metadata, splits, overwrite)
    return {name: len(indices) for name, indices in splits.items()}

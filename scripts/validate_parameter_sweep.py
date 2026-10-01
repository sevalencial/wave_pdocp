import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wave_pdocp.io.dataset_io import read_dataset


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Auditar persistencia, índices y estados guardados; no vuelve a resolver el FOM"
    )
    parser.add_argument(
        "--dataset", type=Path, default=root / "data/datasets/wave_pdocp_fom.h5"
    )
    parser.add_argument(
        "--metrics", type=Path, default=root / "outputs/metrics/fom_parameter_sweep.csv"
    )
    args = parser.parse_args()
    samples, metadata, splits = read_dataset(args.dataset)
    table = pd.read_csv(args.metrics).set_index("sample_id")
    np.testing.assert_array_equal(
        np.sort(samples["sample_id"]), np.sort(table.index[table.status == "PASS"])
    )
    rows = table.loc[samples["sample_id"]]
    if not np.array_equal(rows.status.to_numpy(), samples["status"]):
        raise ValueError("Los estados de CSV y HDF5 no coinciden")
    np.testing.assert_allclose(
        rows.mu.to_numpy(), samples["mu"][:, 0], rtol=1e-14, atol=1e-15
    )
    print(
        f"Persistencia verificada: {len(rows)} muestras, N={metadata['grid_size']}, splits={ {k: len(v) for k, v in splits.items()} }"
    )


if __name__ == "__main__":
    main()

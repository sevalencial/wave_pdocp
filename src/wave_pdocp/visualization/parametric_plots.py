from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_representative_final_states(
    samples, grid, selected, output_path: Path
) -> None:
    """Compare stored controlled states with their targets, separately from adjoints."""
    snapshot_index = {int(i): j for j, i in enumerate(samples["sample_id"])}
    indices = [
        snapshot_index[i]
        for i in selected
        if i in snapshot_index and samples["status"][snapshot_index[i]] != "FAIL"
    ]
    if not indices:
        return

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, constrained_layout=True
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(indices)))
    for k, (ax, target_name, ylabel) in enumerate(
        zip(
            axes,
            ("target_displacement", "target_velocity"),
            (r"Desplazamiento $y_\mu^*(T,x)$", r"Velocidad $v_\mu^*(T,x)$"),
            strict=True,
        )
    ):
        targets = samples[target_name][indices]
        common_target = np.all(targets == targets[0])
        target_symbol = "y" if k == 0 else "v"
        if common_target:
            ax.plot(
                grid.interior_points,
                targets[0],
                "--",
                color="#252525",
                lw=2.4,
                zorder=5,
                label=rf"Target común ${target_symbol}^T(x)$",
            )
        for j, color in zip(indices, colors, strict=True):
            mu_label = f"{samples['mu'][j, 0]:.4f}"
            ax.plot(
                grid.interior_points,
                samples["final_state"][j, k * grid.n : (k + 1) * grid.n],
                color=color,
                lw=1.7,
                label=rf"Estado $\mu={mu_label}$",
            )
            if not common_target:
                ax.plot(
                    grid.interior_points,
                    samples[target_name][j],
                    color=color,
                    ls="--",
                    lw=2,
                    label=rf"Target $\mu={mu_label}$",
                )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if k == 0:
            ax.legend(ncols=3 if common_target else 2, fontsize=8, loc="lower center")
        else:
            ax.legend(
                ncols=3 if common_target else 2,
                fontsize=8,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.16),
            )
    axes[1].set_xlabel("x")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.suptitle(
        "Estados finales bajo control frente al target\nParámetros representativos",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_campaign(table, samples, grid, refinement, selected, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("adjoint_norm", r"$\|P^T\|_{H_h}$", False),
        ("terminal_error", "Error terminal en H_h", False),
        ("control_energy", "Energía del control", False),
        ("optimal_cost", "Costo óptimo", False),
        ("gramian_condition_number", "Condición del Gramiano en H_h", True),
        ("system_condition_number", "Condición del sistema en H_h", True),
        ("gramian_min_eigenvalue", "Autovalor mínimo del Gramiano", False),
        ("state_spectral_abscissa", "Máxima parte real de espectro de A", False),
        (
            "local_parameter_lipschitz_indicator",
            "Sensibilidad local del adjunto",
            False,
        ),
        ("relative_snapshot_change", "Cambio relativo entre adjuntos", False),
        ("relative_residual", "Residual algebraico relativo", True),
        ("optimality_relative_error", "Error relativo de optimalidad", True),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(14, 13), constrained_layout=True)
    for ax, (key, title, logarithmic) in zip(axes.flat, specs, strict=True):
        values = table[key].to_numpy(dtype=float).copy()
        values[~np.isfinite(values)] = np.nan
        ax.plot(table.mu, values, color="#286db0", lw=1.6)
        for status, color, marker in [
            ("WARNING", "#d27c13", "o"),
            ("FAIL", "#ba3030", "x"),
        ]:
            mask = table.status == status
            ax.scatter(
                table.loc[mask, "mu"], values[mask], color=color, marker=marker, s=28
            )
        if logarithmic and np.any(values > 0):
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$\mu$")
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)
    counts = table.status.value_counts()
    fig.suptitle(
        f"Barrido de amplitud de c(x) — PASS {counts.get('PASS', 0)}, WARNING {counts.get('WARNING', 0)}, FAIL {counts.get('FAIL', 0)}",
        fontsize=15,
    )
    fig.savefig(output_dir / "diagnostics_vs_mu.png", dpi=160)
    plt.close(fig)

    if len(samples["sample_id"]) > 0:
        snapshot_index = {int(i): j for j, i in enumerate(samples["sample_id"])}
        fields = np.full((len(table), 2 * grid.n), np.nan)
        for j, row in table.iterrows():
            if row.sample_id in snapshot_index and row.status != "FAIL":
                fields[j] = samples["final_adjoint"][snapshot_index[row.sample_id]]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for k, ax in enumerate(axes):
            data = fields[:, k * grid.n : (k + 1) * grid.n]
            finite = data[np.isfinite(data)]
            vmax = max(float(np.max(np.abs(finite))) if finite.size else 0.0, 1e-15)
            mesh = ax.pcolormesh(
                grid.interior_points,
                table.mu,
                np.ma.masked_invalid(data),
                shading="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            fig.colorbar(mesh, ax=ax, label=f"p{k + 1}")
            ax.set(
                xlabel="x",
                ylabel=r"$\mu$",
                title=f"Componente p{k + 1} del adjunto terminal",
            )
        fig.savefig(output_dir / "adjoint_manifold.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(
            2, 1, figsize=(9, 7), sharex=True, constrained_layout=True
        )
        for i, color in zip(
            selected,
            plt.cm.viridis(np.linspace(0.05, 0.95, len(selected))),
            strict=True,
        ):
            if i not in snapshot_index:
                continue
            j = snapshot_index[i]
            for k, ax in enumerate(axes):
                ax.plot(
                    grid.interior_points,
                    samples["final_adjoint"][j, k * grid.n : (k + 1) * grid.n],
                    color=color,
                    label=f"mu={samples['mu'][j, 0]:.4f}",
                )
                ax.set_ylabel(f"p{k + 1}")
                ax.grid(alpha=0.2)
        axes[0].legend(ncols=2, fontsize=8)
        axes[1].set_xlabel("x")
        fig.suptitle("Adjuntos de los parámetros representativos")
        fig.savefig(output_dir / "representative_adjoints.png", dpi=160)
        plt.close(fig)

        plot_representative_final_states(
            samples,
            grid,
            selected,
            output_dir / "representative_final_states_vs_target.png",
        )

    if not refinement.empty:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for _, rows in refinement.groupby("sample_id"):
            valid = rows.dropna(subset=["relative_adjoint_difference"])
            if not valid.empty:
                ax.loglog(
                    valid.n_interior,
                    valid.relative_adjoint_difference,
                    "o-",
                    label=f"mu={valid.mu.iloc[0]:.4f}",
                )
        ax.set(
            xlabel="N de la malla fina",
            ylabel="Diferencia relativa del adjunto en H_h",
            title="Refinamiento espacial: interpolación gruesa → fina",
        )
        ax.legend(ncols=2, fontsize=8)
        ax.grid(alpha=0.25, which="both")
        fig.savefig(output_dir / "mesh_refinement.png", dpi=160)
        plt.close(fig)

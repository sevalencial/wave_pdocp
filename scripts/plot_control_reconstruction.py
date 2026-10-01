from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from wave_pdocp.domain.grid import SpatialGrid
from wave_pdocp.fom.reconstruction import reconstruct_control, reconstruct_final_state
from wave_pdocp.fom.solution import FOMSolution
from wave_pdocp.io.reports import save_fom_report
from wave_pdocp.orchestration.experiment import SingleMuExperiment

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs/single_mu_variable"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#0d366b", "#f0efec", "#8f1f1f"]
)


def plot_final_adjoint(grid: SpatialGrid, P_T: np.ndarray, out_path: str) -> None:
    n = grid.n
    x = grid.interior_points
    p, q = P_T[:n], P_T[n:]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, p, color=BLUE, lw=2, label=r"$P_T$ (desplazamiento)")
    ax.plot(x, q, color=ORANGE, lw=2, label=r"$P_T$ (velocidad)")
    ax.axhline(0, color="#c3c2b7", lw=1, zorder=0)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$P_T$")
    ax.set_title("Estado adjunto terminal $P_T$")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_control(
    grid: SpatialGrid, t: np.ndarray, U: np.ndarray, out_path: str
) -> None:
    x = grid.interior_points
    vmax = np.max(np.abs(U))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    mesh = ax.pcolormesh(x, t, U, shading="auto", cmap=DIVERGING, vmin=-vmax, vmax=vmax)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(r"Control reconstruido $u(t,x) = -B_{adj}\,e^{(T-t)A_{adj}}\,P_T$")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("u(t, x)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_final_state(solution: FOMSolution, out_path: str) -> None:
    """Estado alcanzado Y(T) = S_T @ Y0 - W @ P_T, comparado contra el target."""
    grid = solution.grid
    fields = solution.fields
    n = grid.n
    x = grid.interior_points

    Y_achieved = reconstruct_final_state(solution)
    y_achieved, v_achieved = Y_achieved[:n], Y_achieved[n:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.plot(x, fields.target_displacement, color=ORANGE, lw=2, ls="--", label="target")
    ax1.plot(x, y_achieved, color=BLUE, lw=2, label="estado bajo control")
    ax1.set_ylabel("desplazamiento")
    ax1.legend(frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(x, fields.target_velocity, color=ORANGE, lw=2, ls="--", label="target")
    ax2.plot(x, v_achieved, color=BLUE, lw=2, label="estado bajo control")
    ax2.set_xlabel("x")
    ax2.set_ylabel("velocidad")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        f"Estado final vs. target — T={solution.final_time:g}, beta={solution.beta:g}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    error = Y_achieved - np.concatenate(
        [fields.target_displacement, fields.target_velocity]
    )
    print(f"||Y(T) - Y_T||_max = {np.max(np.abs(error)):.3e}")


def plot_coefficients(solution: FOMSolution, out_path: str) -> None:
    grid = solution.grid
    fields = solution.fields
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(
        grid.half_points, fields.wave_speed_half, color=BLUE, label="c(x), semipuntos"
    )
    axes[0].set_ylabel("velocidad c(x)")
    axes[1].plot(
        grid.interior_points, fields.damping, color=ORANGE, label="d(x), nodos"
    )
    axes[1].set_ylabel("amortiguamiento d(x)")
    axes[1].set_xlabel("x")
    for ax in axes:
        ax.set_xlim(0.0, grid.length)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Coeficientes espaciales del caso evaluado")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    experiment = SingleMuExperiment.variable_coefficient_example()
    solution = experiment.run()
    grid = solution.grid
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert experiment.diagnostics is not None
    save_fom_report(solution, experiment.diagnostics, OUT_DIR / "metrics.json")

    t, U = reconstruct_control(solution)

    plot_final_adjoint(grid, solution.final_adjoint, f"{OUT_DIR}/p_t.png")
    plot_control(grid, t, U, f"{OUT_DIR}/control_u.png")
    plot_final_state(solution, f"{OUT_DIR}/final_state_vs_target.png")
    plot_coefficients(solution, f"{OUT_DIR}/coefficients.png")

    print(f"P_T guardado en {OUT_DIR}/p_t.png")
    print(f"u(t,x) guardado en {OUT_DIR}/control_u.png")
    print(f"Y(T) vs target guardado en {OUT_DIR}/final_state_vs_target.png")
    print(f"Coeficientes guardados en {OUT_DIR}/coefficients.png")


if __name__ == "__main__":
    main()

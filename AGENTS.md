# AGENTS.md

## Project context

This repository implements the numerical and machine-learning pipeline
for a master's thesis on parametric optimal control of the 1D wave equation.

Before making substantial changes, read:

- `docs/WAVE_PDOCP_CONTEXT.md`

That document contains the mathematical formulation, numerical decisions,
software architecture, and current project state.

## Critical mathematical constraints

- Do not discretize time unless explicitly requested.
- The FOM is spatially semi-discrete and continuous in time.
- Use weighted adjoints with `Q_h` and `R_h`; do not replace them by plain transposes.
- The terminal-adjoint equation is

  `(I + beta W_mu,h(T)) P_mu,h^T
   = beta (exp(T A_mu,h) Y_mu,h^0 - Y_mu,h^T)`.

- Do not compute explicit matrix inverses when a linear solve can be used.
- For variable `c_mu(x)`, discretize the divergence-form operator
  `d/dx(c_mu^2(x) dy/dx)` conservatively using half-grid coefficients.

## Architecture

- OOP for stateful components and orchestration.
- Functional style for mathematical operators.
- Keep physics, discretization, operators, FOM, diagnostics,
  visualization, reduction, certification, and surrogates separated.
- `FOMSolver` must not become a monolithic class.
- The orchestrator coordinates workflows but must not contain mathematical formulas.

## Current priority

For the current development milestone, read:
- `docs/PARAMETER_SWEEP_FOM_TASKS.md`

## Validation

After numerical changes, run:

```bash
pytest
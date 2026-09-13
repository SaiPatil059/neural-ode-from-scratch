# Neural ODE from Scratch

Here I will be trying to build the Runge-Kutta 4th order numerical method from scratch and apply it to forced and free damped mechanical vibration systems, validated against analytical solutions, then use it as the core integrator for a Neural ODE.

**Note on tools:** Stage 1 (RK4 validation) is implemented in MATLAB, consistent with coursework in Numerical Methods and Mechanical Vibrations. Stage 2 (Neural ODE) is implemented in Python/PyTorch, since it involves training a neural network — the RK4 solver and analytical solution were re-ported and re-validated in Python before being used in Stage 2.

---

## Stage 1: RK4 Validation (MATLAB) — Complete!

Implemented a 4th-order Runge-Kutta integrator from scratch for a damped single-degree-of-freedom vibration system: mẍ + cẋ + kx = F(t).

- Validated against the closed-form analytical solution across all three damping regimes: underdamped (ζ<1), critically damped (ζ=1), and overdamped (ζ>1) — free decay case.
- Extended to forced vibration (F(t) = F₀cos(ωt)), validated against the full analytical solution (transient + steady-state).
- Verified numerical error decays as expected with decreasing step size: measured convergence order ≈ **4.01**, matching RK4's theoretical 4th-order accuracy (log-log error vs. step size, five step sizes from 1e-2 to 1e-4).
- Along the way, found and fixed a stale-variable bug (a leftover workspace variable silently reused across script edits) that had been inflating numerical-vs-analytical error by ~8 orders of magnitude — caught by checking error magnitude against theoretical expectation rather than trusting a visually-overlapping plot.

**Files:** `main.mlx`

## Stage 2: Neural ODE Wrapper (Python/PyTorch) — Complete!

Re-implemented and re-validated the RK4 solver and analytical solution in Python/NumPy (max error ~1.1e-9 vs. the known solution, matching the MATLAB result) before building on it.

Built a residual/hybrid Neural ODE: `f_θ(t, X) = f_known(t, X) + correction_net(t, X)`, where `f_known` is the exact physics from Stage 1 and `correction_net` is a small PyTorch network. Trained on the ground-truth trajectory at a coarser step size (dt=0.01) for tractable training time, then validated at the original fine resolution (dt=0.001) the network never saw during training.

- Training loss decreases smoothly and converges (see `errorplot_python.png`).
- Trained model's trajectory closely tracks the known-physics/analytical solution at fine resolution: max error ≈ **0.017**, compared to ≈1e-9 for pure known-physics RK4 — roughly 0.3–0.5% of the trajectory's signal range (~-3.2 to 5).
- Result is reproducible across independent training runs (fresh random initialization), producing similar loss curves and final error.

**Files:** `neural_ode_from_scratch.ipynb`

## Stage 3: Noise Robustness — In progress...

Planned: inject noise into the observed trajectory or system dynamics, compare fixed-step RK4 against an adaptive-step method (e.g., RK45), and characterize where/why the fixed-step solver degrades under noise.

---

## Repo structure
- `main.mlx` — Stage 1, MATLAB
- `neural_ode_from_scratch.py` — Stage 2, Python
- plots/ — validation and error plots referenced above

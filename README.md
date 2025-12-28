# Robotics Autonomy Sim

A lightweight 2D autonomy sandbox for **non-holonomic mobile robots**, built to deeply understand
**planning → control → plant → sensing → estimation** as a closed-loop system.

This project starts intentionally simple (Python + Matplotlib, kinematic diff-drive),
with the goal of growing toward higher-fidelity simulation, comparative controller studies,
and eventual C++ / Gazebo / real-robot deployment.

---

## Motivation

Most autonomy stacks are learned *top-down* through large frameworks.
This repo is the opposite: a **from-scratch, geometry-first** environment to build intuition for:

- Path planning on grid maps
- Geometric path tracking (Pure Pursuit today; Stanley/MPC later)
- Discrete-time closed-loop behavior
- Timing, latency, and controller stability
- Clean modular interfaces that scale with fidelity

The emphasis is **learning and insight**, not production code.

---

## Current Features (MVP)

- 2D grid world with start/goal
- Global path planning (placeholder → A*)
- Differential-drive kinematic plant
- **Pure Pursuit** path tracker
- Fixed-step discrete-time simulation
- Matplotlib visualization with:
  - robot footprint & heading
  - path + trail
  - lookahead geometry (optional)
  - pause / step / browse history
- Optional CSV logging for clean, offline analysis

---

## Run
Basic:
```bash
python3 -m scripts.run_simulation
```
With detailed visualization overlays (lookahead + debug):
```bash
python3 -m scripts.run_simulation --visualize_detailed
```
With CSV logging (saved for offline plots/analysis):
```bash
python3 -m scripts.run_simulation --log_csv logs/pp_run.csv
```

Detailed + logging:
```bash
python3 -m scripts.run_simulation --visualize_detailed --log_csv logs/<xyz>.csv
```
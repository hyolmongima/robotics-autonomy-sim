# Robotics Autonomy Sim
A lightweight 2D autonomy stack for non-holonomic mobile robots, built for personal exploration.

The project begins intentionally simple, with a basic Python-based kinematic simulation for planning and control. Over time, it is designed to scale toward higher-fidelity simulation, comparative evaluation of algorithms across planning, control, and estimation, and eventual deployment in C++ with Gazebo and on real robots.

## Current Features (Growing)
- Global path planning with A*
- Pure Pursuit Path tracker
- Differential-drive kinematic plant

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

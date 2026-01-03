# Physics Simulation Workbench

Foundation for a physics-first, headless simulation core.

## Developer Quickstart

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python -m bean_physics.app
```

RB1 smoke test scenario:

- Open `examples/scenarios/rigid_body_spin_box_v1.json` in the app (File -> Load).
- Use the Objects panel to select the rigid body, adjust size/mass/position, and Apply.

Visual models (RB3) are rendering-only and do not affect physics.

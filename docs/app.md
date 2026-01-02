# Desktop App (Phase A)

Install UI extras:

```bash
pip install -e ".[ui]"
```

Run the app:

```bash
python -m bean_physics.app
```

Usage:

- Click `Load` and select a scenario JSON (for example `examples/scenarios/two_body_orbit_v1.json`).
- Click `Run` to start playback, `Pause` to stop, `Step` for a single tick, and `Reset` to return to the initial state.
- Use `New` for a blank scenario (marked dirty until you `Save`/`Save As`).
- Use `Save` or `Save As` from the toolbar or File menu. When there are unsaved changes, you will be prompted before `New` or `Load`.

Editing particles:

- Use the `Objects` dock to select particles; double-click a particle to open the Object Details inspector.
- Edit position/velocity/mass in the inspector and click `Apply` to write changes and reset the simulation state.
- Edits mark the session as dirty; remember to `Save` or `Save As` to persist them.

Forces:

- Use the `Objects` dock buttons to add Uniform Gravity or N-body Gravity.
- Select a force and double-click to edit its parameters in the Object Details inspector.
- Clicking `Apply` updates the scenario, resets the simulation, and marks the session dirty.

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

- Use the `Particles` dock to add or remove rows and edit x/y/z, vx/vy/vz, and mass.
- Click `Apply` to write changes to the scenario and reset the simulation state.
- Edits mark the session as dirty; remember to `Save` or `Save As` to persist them.

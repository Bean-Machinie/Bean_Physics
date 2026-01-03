# Scenario Schema v1

This schema defines a declarative JSON format for reproducible simulations.

## Top-level fields

- `schema_version` (int, required): must be `1`.
- `metadata` (object, optional):
  - `name` (string, optional)
  - `description` (string, optional)
- `simulation` (object, required):
  - `dt` (float, required, > 0)
  - `steps` (int, required, >= 0)
  - `integrator` (string, required): `"symplectic_euler"` or `"velocity_verlet"`
- `sampling` (object, optional):
  - `every` (int, optional, > 0) number of steps between samples
- `entities` (object, optional):
  - `particles` (object, optional):
    - `pos` (array[N][3])
    - `vel` (array[N][3])
    - `mass` (array[N])
  - `rigid_bodies` (object, optional):
    - `pos` (array[M][3])
    - `vel` (array[M][3])
    - `quat` (array[M][4], body->world, `[w,x,y,z]`)
    - `omega_body` (array[M][3], body frame)
    - `mass` (array[M])
    - `mass_distribution` (object, required):
      - `points_body` (array[K][3])
      - `point_masses` (array[K])
      - `inertia_body` (array[3][3] or array[M][3][3], optional)
    - `source` (array[M], optional template metadata):
      - `kind` ("box", "sphere", or "points")
      - `params` (`{"size":[sx,sy,sz]}` or `{"radius": r}`)
      - `points` (`[{mass, pos}]`, required when `kind = "points"`)
      - `mass` (float, should match `mass[i]`)

## Models/forces

`models` is a list composed in order; each item is an object with one key:

- `uniform_gravity`:
  - `g` (array[3])
- `nbody_gravity`:
  - `G` (float)
  - `softening` (float, optional)
  - `chunk_size` (int or null, optional)
- `rigid_body_forces`:
  - `forces` (array of objects):
    - `body_index` (int)
    - `point_body` (array[3]) point in body coordinates relative to CoM
    - `force_body` (array[3]) force in body coordinates
    - `enabled` (bool, optional; defaults true)

Legacy note: older scenarios may use `force_world`. These are interpreted as
body-fixed thrusters aligned with the body at t=0 (converted using the initial
orientation on load).

## Notes

- Scenario data is declarative; runtime objects are built by adapters.
- All arrays are interpreted as float64.

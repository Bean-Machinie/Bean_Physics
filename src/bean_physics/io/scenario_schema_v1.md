# Scenario Schema v1

This schema defines a declarative JSON format for reproducible simulations.

## Top-level fields

- `schema_version` (int, required): must be `1`.
- `units` (object, optional):
  - `preset` (string): `"SI"`, `"KM"`, or `"ASTRO"`
  - `enabled` (bool): when `false`, values are treated as SI without conversion
- `metadata` (object, optional):
  - `name` (string, optional)
  - `description` (string, optional)
  - `mission_analysis` (object, optional): UI metadata such as Hohmann planner inputs
- `simulation` (object, required):
  - `dt` (float, required, > 0)
  - `steps` (int, required, >= 0)
  - `integrator` (string, required): `"symplectic_euler"` or `"velocity_verlet"`
- `sampling` (object, optional):
  - `every` (int, optional, > 0) number of steps between samples
- `impulse_events` (array, optional):
  - `t` (float, required, >= 0): simulation time for the impulse
  - `target` (string, required): id in `entities.particles.ids` or `entities.rigid_bodies.ids`
  - `delta_v_world` (array[3], required): instantaneous velocity change in world frame
  - `label` (string, optional)
- `entities` (object, optional):
  - `particles` (object, optional):
    - `pos` (array[N][3])
    - `vel` (array[N][3])
    - `mass` (array[N])
    - `ids` (array[N], optional): unique string ids for particles
    - `visual` (array[N], optional):
      - `null` or visual block (see below)
  - `rigid_bodies` (object, optional):
    - `pos` (array[M][3])
    - `vel` (array[M][3])
    - `quat` (array[M][4], body->world, `[w,x,y,z]`)
    - `omega_body` (array[M][3], body frame)
    - `mass` (array[M])
    - `ids` (array[M], optional): unique string ids for rigid bodies
    - `mass_distribution` (object, required):
      - `points_body` (array[K][3])
      - `point_masses` (array[K])
      - `inertia_body` (array[3][3] or array[M][3][3], optional)
    - `source` (array[M], optional template metadata):
      - `kind` ("box", "sphere", or "points")
      - `params` (`{"size":[sx,sy,sz]}` or `{"radius": r}`)
      - `points` (`[{mass, pos}]`, required when `kind = "points"`)
      - `mass` (float, should match `mass[i]`)
    - `visual` (array[M], optional):
      - `null` or visual block (see below)

## Visual blocks

Visual blocks are UI-only and do not affect physics.

- `kind` (string): `"mesh"` or `"primitive"`
- `mesh_path` (string, required for `"mesh"`)
- `scale` (array[3], optional; default `[1,1,1]`)
- `offset_body` (array[3], optional; default `[0,0,0]`)
- `rotation_body_quat` (array[4], optional; default `[1,0,0,0]`)
- `color_tint` (array[3], optional; default `[1,1,1]`, multiplies authored colors)

Meshes are recentered to their geometric center for rendering.

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
    - `throttle` (float, optional; defaults 1.0, [0..1])
    - `name` (string, optional)
    - `group` (string, optional)
    - `enabled` (bool, optional; defaults true)

Legacy note: older scenarios may use `force_world`. These are interpreted as
body-fixed thrusters aligned with the body at t=0 (converted using the initial
orientation on load).

## Notes

- Scenario data is declarative; runtime objects are built by adapters.
- All arrays are interpreted as float64.
- Runtime physics always operate in SI units (m, kg, s); scenario values are stored
  in the selected preset units when `units.enabled` is true.

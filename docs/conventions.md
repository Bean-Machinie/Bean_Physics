# Conventions

## Quaternions

- Storage order: q = [w, x, y, z].
- Meaning: q represents a rotation from BODY to WORLD (body->world).
- Vector rotation: _world = R(q) * v_body.
- Multiplication: Hamilton product.
- Composition: applying q1 then q2 is q = quat_mul(q2, q1).

## Angular velocity

- Angular velocity is expressed in the WORLD frame (omega_world) unless otherwise stated.

## Orientation integration

- The integrator uses the exponential map to build a delta quaternion from omega * dt.
- For world-frame angular velocity with body->world quaternions, the update is:
  q_next = quat_mul(delta_q, q).
- Quaternions are renormalized to reduce drift.
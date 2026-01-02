# Conventions

## Quaternions

- Storage order: q = [w, x, y, z].
- Meaning: q represents a rotation from BODY to WORLD (body->world).
- Vector rotation: v_world = R(q) * v_body.
- Multiplication: Hamilton product.
- Composition: applying q1 then q2 is q = quat_mul(q2, q1).

## Angular velocity

- Particles: not applicable.
- Rigid bodies: angular velocity is stored in the BODY frame (omega_body).

## Orientation integration

- The integrator uses the exponential map to build a delta quaternion from omega * dt.
- For body-frame angular velocity with body->world quaternions, the update is:
  q_next = quat_mul(q, delta_q).
- (Migration note) Earlier phases used omega in WORLD frame with left-multiply updates.
- Quaternions are renormalized to reduce drift.

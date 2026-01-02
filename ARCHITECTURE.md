# Architecture

## Layering rules

- core is pure simulation logic and must remain headless (no UI, no rendering, no file dialogs).
- io handles loading/saving scenarios and results; it may depend on core but not on iz.
- iz is optional and may depend on core but never the other way around.

## State representation (intent)

- Use explicit, strongly typed dataclasses for state (positions, velocities, masses, time).
- Keep state immutable between steps where practical; step functions return new state.
- Deterministic fixed-step integration is the default; all randomness is seedable and injected.

## Extension points

- **Forces**: composable functions/objects that compute accelerations from state.
- **Integrators**: fixed-step integrators (e.g., Euler, Verlet) that advance state.
- **Constraints**: optional projection/constraint systems applied post-integration.

## Reproducibility

- Fixed time step and explicit seeds for any stochastic elements.
- No global RNG usage inside core; RNG passed via parameters.
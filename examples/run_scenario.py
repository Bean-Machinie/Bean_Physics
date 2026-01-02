"""Run a scenario JSON and optionally save sampled data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bean_physics.core.diagnostics import (
    kinetic_energy,
    linear_momentum,
    total_energy_gravity,
    total_mass,
)
from bean_physics.core.run import run
from bean_physics.io import load_scenario, scenario_to_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    defn = load_scenario(args.scenario)
    state, model, integrator, dt, steps, aux = scenario_to_runtime(defn)
    sample_every = defn.get("sampling", {}).get("every")

    result = run(state, model, integrator, dt, steps, sample_every=sample_every)
    final = result.final_state

    print("steps:", steps)
    print("dt:", dt)
    print("sim time:", dt * steps)
    if final.particles is not None:
        print("particles total mass:", total_mass(final))
        print("particles momentum:", linear_momentum(final))
        print("particles KE:", kinetic_energy(final))
        if any("nbody_gravity" in m for m in defn.get("models", [])):
            gval = None
            for entry in defn.get("models", []):
                if "nbody_gravity" in entry:
                    gval = entry["nbody_gravity"]["G"]
            if gval is not None:
                print("particles total energy:", total_energy_gravity(final, G=gval))

    if args.out is not None and result.time is not None:
        np.savez_compressed(
            args.out,
            time=result.time,
            particles_pos=result.particles_pos,
            particles_vel=result.particles_vel,
            rigid_pos=result.rigid_pos,
            rigid_quat=result.rigid_quat,
            rigid_omega_body=result.rigid_omega_body,
        )
        print("saved samples to:", args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

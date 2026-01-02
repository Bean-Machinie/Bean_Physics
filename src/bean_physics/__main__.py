"""CLI entrypoint placeholder."""

from __future__ import annotations

from . import __version__


def main() -> int:
    print(f"bean_physics v{__version__}")
    print("examples: <none>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
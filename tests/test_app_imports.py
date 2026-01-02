from __future__ import annotations

import pytest


pytest.importorskip("PySide6")
pytest.importorskip("vispy")


def test_app_modules_import_without_launching() -> None:
    import bean_physics.app  # noqa: F401
    import bean_physics.app.main  # noqa: F401
    import bean_physics.app.viewport  # noqa: F401
    import bean_physics.app.window  # noqa: F401

from __future__ import annotations


def test_sanity_import() -> None:
    import bean_physics as bp
    import numpy as np

    assert isinstance(bp.__version__, str)
    assert np.add(1.0, 2.0) == 3.0
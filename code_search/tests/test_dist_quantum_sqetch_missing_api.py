"""The sample_low_weight_logicals_sqetch wrapper targets an API that the
sqetch package does not provide.

core/dist/quantum_sqetch.py wraps sqetch.sample_low_weight_logicals (the
DistRandTailored analog needed by adaptive-thickening surgery), but the
sqetch package ships no function of that name, so every call path through
the wrapper fails at attribute lookup before any GPU work. These tests pin
the absence and the resulting failure mode so a future sqetch release that
adds the API flips them loudly.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.fast


def test_sqetch_package_lacks_sample_low_weight_logicals():
    sqetch = pytest.importorskip("sqetch")
    assert not hasattr(sqetch, "sample_low_weight_logicals")


def test_wrapper_raises_attribute_error_before_any_gpu_work():
    pytest.importorskip("sqetch")
    from core.dist.quantum_sqetch import sample_low_weight_logicals_sqetch

    # Steane-code shapes: valid inputs, so the failure is the missing API,
    # not input validation
    H = np.array([[0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
    L = np.array([[1, 1, 1, 0, 0, 0, 0]], dtype=np.uint8)
    with pytest.raises(AttributeError, match="sample_low_weight_logicals"):
        sample_low_weight_logicals_sqetch(
            H, L, num_trials=10, target_weight=3, max_results=4)

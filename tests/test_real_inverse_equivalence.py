"""Real (non-mocked) numerical equivalence between chunked and unchunked
raw inverse label extraction.

Every prior review round in the issue #271 correspondence trail (see
plans/correspondance/) declined to claim scientific equivalence because only
mocked source estimates were available. This test exercises the actual
production code path (``SequentialProcessor._extract_raw_label_time_course``)
against a real fsaverage/DK-atlas inverse operator, at Wave's requested
tolerance (correspondence 002-wave-raw-chunk-scientific-equivalence-consultation.html).

Opt-in: requires network access to fetch fsaverage (~430MB, cached after the
first run) and is skipped unless ``RUN_REAL_INVERSE_TESTS=1`` is set, to keep
default test runs fast and independent of network/CA-bundle availability.
"""

import os

import numpy as np
import pytest

from _real_inverse_fixtures import build_raw_fixture, make_processor

pytestmark = pytest.mark.real_inverse

REQUIRED_ENV = "RUN_REAL_INVERSE_TESTS"


@pytest.fixture(scope="module")
def real_fixture():
    if os.environ.get(REQUIRED_ENV) != "1":
        pytest.skip(f"set {REQUIRED_ENV}=1 to run real-inverse release-evidence tests")
    try:
        return build_raw_fixture(n_seconds=300.0, sfreq=250.0)
    except Exception as exc:  # network, CA-bundle, or dataset unavailable
        pytest.skip(f"real fsaverage fixture unavailable: {exc}")


def test_chunked_matches_unchunked_exactly_on_real_inverse(real_fixture):
    raw, inv, labels, src = real_fixture

    # A chunk_seconds longer than the recording is one whole-Raw chunk,
    # i.e. the unchunked oracle computed through the same chunked code path.
    unchunked = make_processor(chunk_seconds=raw.times[-1] + 1.0, src=src, labels=labels)
    chunked = make_processor(chunk_seconds=30.0, src=src, labels=labels)

    ts_unchunked = unchunked._extract_raw_label_time_course(raw, inv)
    ts_chunked = chunked._extract_raw_label_time_course(raw, inv)

    assert ts_chunked.shape == ts_unchunked.shape
    np.testing.assert_allclose(ts_chunked, ts_unchunked, rtol=1e-10, atol=1e-15, equal_nan=True)

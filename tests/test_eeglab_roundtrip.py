"""EEGLAB round-trip release-evidence gate for issue #271.

Exports chunked raw inverse output through the real ``.set`` writer
(``SequentialProcessor._convert_raw_inverse_to_eeg``) and reads it back with
``mne.io.read_raw_eeglab`` to prove serialization does not corrupt or
misalign the chunked result. Compared at a float32-appropriate tolerance
(EEGLAB's ``.set``/``.fdt`` format stores data as float32), not the strict
float64 in-memory oracle used by test_real_inverse_equivalence.py -- Wave's
advisory explicitly scoped that stricter tolerance to in-memory comparison
because file serialization introduces its own precision effect.

Opt-in: requires network access to fetch fsaverage (~430MB, cached after the
first run) and is skipped unless ``RUN_REAL_INVERSE_TESTS=1`` is set.
"""

import os

import mne
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
        return build_raw_fixture(n_seconds=60.0, sfreq=250.0)
    except Exception as exc:  # network, CA-bundle, or dataset unavailable
        pytest.skip(f"real fsaverage fixture unavailable: {exc}")


def test_chunked_output_round_trips_through_eeglab_export(real_fixture, tmp_path):
    raw, inv, labels, src = real_fixture
    processor = make_processor(chunk_seconds=15.0, src=src, labels=labels)

    out_raw, out_path = processor._convert_raw_inverse_to_eeg(
        raw, inv, str(tmp_path), subject_id="synthtest"
    )
    assert os.path.exists(out_path)

    pre_export_data = out_raw.get_data()

    roundtrip = mne.io.read_raw_eeglab(out_path, preload=True, verbose=False)
    rt_data = roundtrip.get_data()

    assert roundtrip.ch_names == out_raw.ch_names
    assert roundtrip.info["sfreq"] == out_raw.info["sfreq"]
    assert rt_data.shape == pre_export_data.shape

    np.testing.assert_array_equal(roundtrip.annotations.onset, out_raw.annotations.onset)
    np.testing.assert_array_equal(roundtrip.annotations.duration, out_raw.annotations.duration)
    assert list(roundtrip.annotations.description) == list(out_raw.annotations.description)

    # float32 storage tolerance, not the strict float64 in-memory oracle.
    np.testing.assert_allclose(rt_data, pre_export_data, rtol=1e-5, atol=1e-8)

"""Peak-RSS release-evidence gate for issue #271.

Proves that chunked raw inverse solving bounds peak process memory and does
not scale with recording length the way the unchunked whole-Raw path does.
Uses a real fsaverage/DK-atlas inverse (see _real_inverse_fixtures.py), run
in isolated subprocesses (via _rss_worker.py) so each measurement reflects a
single clean peak rather than a cumulative one across repeated runs in the
same process.

Thresholds are relative (chunked vs. unchunked, same machine, same run)
rather than absolute MB values, since absolute peak RSS is sensitive to
Python/NumPy/MNE baseline footprint, allocator behavior, and CI noise in a
way relative comparisons are not.

Opt-in: set RUN_REAL_INVERSE_TESTS=1. Not run by default because it is slow
(roughly two minutes) and needs network access to fetch fsaverage on first
use.
"""

import os
import subprocess
import sys
import time

import psutil
import pytest

pytestmark = pytest.mark.real_inverse

REQUIRED_ENV = "RUN_REAL_INVERSE_TESTS"
WORKER = os.path.join(os.path.dirname(__file__), "_rss_worker.py")


def _peak_rss_bytes(n_seconds: float, chunk_seconds: float, poll_interval: float = 0.05) -> int:
    """Run one worker subprocess and return its peak (self + children) RSS."""
    proc = subprocess.Popen(
        [sys.executable, WORKER, str(n_seconds), str(chunk_seconds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    watcher = psutil.Process(proc.pid)
    peak = 0
    while proc.poll() is None:
        try:
            rss = watcher.memory_info().rss
            for child in watcher.children(recursive=True):
                rss += child.memory_info().rss
            peak = max(peak, rss)
        except psutil.NoSuchProcess:
            pass
        time.sleep(poll_interval)
    out, _ = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"RSS worker failed (n_seconds={n_seconds}, chunk_seconds={chunk_seconds}):\n{out}"
        )
    return peak


@pytest.fixture(scope="module")
def require_real_inverse():
    if os.environ.get(REQUIRED_ENV) != "1":
        pytest.skip(f"set {REQUIRED_ENV}=1 to run real-inverse release-evidence tests")
    try:
        # Cheap availability probe; the full per-duration fixture is rebuilt
        # inside each subprocess so every measurement is a clean process.
        from mne.datasets import fetch_fsaverage

        fetch_fsaverage(verbose=False)
    except Exception as exc:
        pytest.skip(f"real fsaverage fixture unavailable: {exc}")


def test_chunked_peak_memory_is_bounded_and_does_not_scale_with_duration(require_real_inverse):
    short_unchunked = _peak_rss_bytes(300.0, 300.0)
    short_chunked = _peak_rss_bytes(300.0, 30.0)
    long_unchunked = _peak_rss_bytes(900.0, 900.0)
    long_chunked = _peak_rss_bytes(900.0, 30.0)

    # Chunking must reduce peak memory substantially at both durations.
    assert short_chunked < short_unchunked * 0.5, (
        f"chunked ({short_chunked/1e9:.2f}GB) should be well below "
        f"unchunked ({short_unchunked/1e9:.2f}GB) at 300s"
    )
    assert long_chunked < long_unchunked * 0.5, (
        f"chunked ({long_chunked/1e9:.2f}GB) should be well below "
        f"unchunked ({long_unchunked/1e9:.2f}GB) at 900s"
    )

    # The chunked path's peak must not scale with recording length the way
    # the unchunked path's does. This allows generous headroom (2x) since it
    # asserts the qualitative shape of the curve (flat vs. growing), not a
    # precise ratio.
    assert long_chunked < short_chunked * 2.0, (
        f"chunked peak grew {long_chunked/short_chunked:.1f}x when duration "
        "only tripled; expected roughly flat memory"
    )

    # The unchunked path is expected to keep growing with duration; this
    # documents the contrast this fix addresses rather than gating on an
    # exact ratio.
    assert long_unchunked > short_unchunked

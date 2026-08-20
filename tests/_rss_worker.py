"""Subprocess worker for test_memory_gate.py.

Runs one variant (chunked or unchunked) of raw inverse label extraction in
its own process so the parent test can measure a clean peak RSS per run
without cross-run allocator/cache contamination. Not a test module: does
not start with ``test_``, so pytest does not collect it, and it is invoked
via ``subprocess``, not imported by test code directly.
"""

import sys

from _real_inverse_fixtures import build_raw_fixture, make_processor


def main():
    n_seconds = float(sys.argv[1])
    chunk_seconds = float(sys.argv[2])
    raw, inv, labels, src = build_raw_fixture(n_seconds=n_seconds, sfreq=250.0)
    processor = make_processor(chunk_seconds=chunk_seconds, src=src, labels=labels)
    label_ts = processor._extract_raw_label_time_course(raw, inv)
    print(f"shape={label_ts.shape}")


if __name__ == "__main__":
    main()

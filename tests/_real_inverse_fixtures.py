"""Helpers for building a real (non-mocked) fsaverage-based inverse fixture.

These are opt-in, network-dependent helpers used by the ``real_inverse``-marked
release-evidence tests (test_real_inverse_equivalence.py, test_memory_gate.py,
test_eeglab_roundtrip.py). They intentionally do not touch the mocked fixtures
in conftest.py used by the rest of the suite.

Not a test module itself: does not start with ``test_`` so pytest does not
collect it, and it is also imported directly by the RSS subprocess worker
(_rss_worker.py), which runs outside of pytest.
"""

import os

import mne
import numpy as np

mne.set_log_level("ERROR")


def build_raw_fixture(n_seconds: float, sfreq: float = 250.0, seed: int = 0):
    """Build a synthetic Raw plus a real fsaverage/DK-atlas inverse operator.

    Uses the same montage (GSN-HydroCel-129), ico-5 source space, BEM, and
    DK-atlas labels the production SequentialProcessor uses, so equivalence,
    memory, and round-trip evidence gathered against this fixture reflects
    the actual scientific configuration rather than a synthetic stand-in.
    Downloads fsaverage (~430MB) via MNE on first use; cached under
    ``MNE_DATA``/``~/mne_data`` afterward.
    """
    from mne.datasets import fetch_fsaverage

    fs_dir = fetch_fsaverage(verbose=False)
    subjects_dir = os.path.dirname(fs_dir)
    src = mne.read_source_spaces(os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif"))
    bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
    labels = mne.read_labels_from_annot("fsaverage", parc="aparc", subjects_dir=subjects_dir)
    labels = [label for label in labels if "unknown" not in label.name]

    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    info = mne.create_info(montage.ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage, match_case=False, on_missing="ignore")

    rng = np.random.default_rng(seed)
    n_samples = int(n_seconds * sfreq)
    data = rng.normal(scale=2e-6, size=(len(info["ch_names"]), n_samples))
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_eeg_reference(projection=True, verbose=False)
    raw.set_annotations(
        mne.Annotations(onset=[1.0, 5.0], duration=[0.5, 0.2], description=["test1", "test2"])
    )

    fwd = mne.make_forward_solution(
        raw.info,
        trans="fsaverage",
        src=src,
        bem=bem,
        eeg=True,
        meg=False,
        mindist=5.0,
        n_jobs=1,
        verbose=False,
    )
    noise_cov = mne.make_ad_hoc_cov(raw.info, verbose=False)
    inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov, verbose=False)
    return raw, inv, labels, src


def make_processor(chunk_seconds, src, labels):
    """Build a SequentialProcessor with a pre-computed src/labels cache.

    Bypasses ``_setup_fsaverage`` (which re-fetches/re-reads from disk) so
    callers can reuse one already-loaded src/labels pair across processors.
    """
    from autoclean_eeg2source.core.converter import SequentialProcessor
    from autoclean_eeg2source.core.memory_manager import MemoryManager

    processor = SequentialProcessor(
        memory_manager=MemoryManager(max_memory_gb=8), chunk_seconds=chunk_seconds
    )
    processor.fsaverage_src = src
    processor.labels = labels
    return processor

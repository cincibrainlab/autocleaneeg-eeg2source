"""Tests for memory-safe continuous/raw source localization chunking."""

import gc
import inspect
import math
import multiprocessing as mp
from types import SimpleNamespace
from unittest.mock import MagicMock
import weakref
from datetime import datetime, timezone

import mne
import numpy as np
import pytest

from autoclean_eeg2source.core.converter import SequentialProcessor
from autoclean_eeg2source.core.gpu_processor import GPUProcessor
from autoclean_eeg2source.core.parallel_processor import CachedProcessor, ParallelProcessor
from autoclean_eeg2source.core.robust_processor import RobustProcessor
from autoclean_eeg2source.utils.benchmarking import run_standard_benchmark
from autoclean_eeg2source import cli


class DummyRaw:
    """Small raw-like object with only the fields chunking needs."""

    def __init__(self, sfreq=10.0, n_times=25):
        self.info = {"sfreq": sfreq}
        self.n_times = n_times
        self.annotations = None
        self.first_samp = 0
        self.data = np.arange(2 * n_times, dtype=float).reshape(2, n_times)


@pytest.fixture
def memory_manager():
    manager = MagicMock()
    manager.check_available.return_value = True
    return manager


def _processor(memory_manager, chunk_seconds=0.5):
    processor = SequentialProcessor(
        memory_manager=memory_manager,
        chunk_seconds=chunk_seconds,
    )
    processor.labels = [
        SimpleNamespace(name="region-lh", pos=np.array([[0.0, 0.0, 0.1]]))
    ]
    processor.fsaverage_src = object()
    return processor


def test_raw_chunk_bounds_uses_half_open_sample_windows(memory_manager):
    processor = _processor(memory_manager, chunk_seconds=0.5)
    raw = DummyRaw(sfreq=10.0, n_times=25)

    assert processor._raw_chunk_bounds(raw) == [
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        (20, 25),
    ]


def test_raw_inverse_chunking_concatenates_label_time_courses(
    monkeypatch, memory_manager
):
    processor = _processor(memory_manager, chunk_seconds=0.5)
    raw = DummyRaw(sfreq=10.0, n_times=25)
    calls = []

    def fake_apply_inverse_raw(raw_arg, inv, **kwargs):
        calls.append((kwargs["start"], kwargs["stop"]))
        n_times = kwargs["stop"] - kwargs["start"]
        return SimpleNamespace(
            data=np.zeros((2, n_times)),
            tstep=1.0 / raw_arg.info["sfreq"],
        )

    def fake_extract_label_time_course(stc, labels, src, mode, verbose):
        return np.full((len(labels), stc.data.shape[1]), len(calls), dtype=float)

    monkeypatch.setattr("mne.minimum_norm.apply_inverse_raw", fake_apply_inverse_raw)
    monkeypatch.setattr("mne.extract_label_time_course", fake_extract_label_time_course)

    label_ts = processor._extract_raw_label_time_course(raw, inv=object())

    assert calls == [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]
    assert label_ts.shape == (1, 25)
    assert memory_manager.check_available.call_count == 5


def test_chunked_result_matches_locked_whole_raw_oracle(monkeypatch, memory_manager):
    processor = _processor(memory_manager, chunk_seconds=0.6)
    raw = DummyRaw(sfreq=10.0, n_times=25)
    inv = object()

    def fake_apply_inverse_raw(raw_arg, inv_arg, **kwargs):
        assert inv_arg is inv
        assert kwargs["lambda2"] == processor.lambda2
        assert kwargs["method"] == "MNE"
        assert kwargs["pick_ori"] == "normal"
        start = kwargs.get("start", 0)
        stop = kwargs.get("stop", raw_arg.n_times)
        return SimpleNamespace(data=raw_arg.data[:, start:stop] * 0.125)

    def fake_extract_label_time_course(stc, labels, src, mode, verbose):
        assert mode == "mean"
        return np.mean(stc.data, axis=0, keepdims=True)

    monkeypatch.setattr("mne.minimum_norm.apply_inverse_raw", fake_apply_inverse_raw)
    monkeypatch.setattr("mne.extract_label_time_course", fake_extract_label_time_course)

    whole_stc = fake_apply_inverse_raw(
        raw,
        inv,
        lambda2=processor.lambda2,
        method="MNE",
        pick_ori="normal",
    )
    locked_whole_raw_oracle = fake_extract_label_time_course(
        whole_stc, processor.labels, processor.fsaverage_src, "mean", False
    )
    chunked = processor._extract_raw_label_time_course(raw, inv)

    np.testing.assert_allclose(
        chunked,
        locked_whole_raw_oracle,
        rtol=1e-10,
        atol=1e-15,
    )


def test_raw_pipeline_reduces_and_releases_each_stc_before_next_chunk(
    monkeypatch, memory_manager
):
    processor = _processor(memory_manager, chunk_seconds=0.5)
    raw = DummyRaw(sfreq=10.0, n_times=12)
    events = []
    previous_stc = None

    class TrackedStc:
        def __init__(self, n_times):
            self.data = np.zeros((2, n_times))

    def fake_apply_inverse_raw(raw_arg, inv, **kwargs):
        nonlocal previous_stc
        gc.collect()
        if previous_stc is not None:
            assert previous_stc() is None
        events.append(("apply", kwargs["start"], kwargs["stop"]))
        stc = TrackedStc(kwargs["stop"] - kwargs["start"])
        previous_stc = weakref.ref(stc)
        return stc

    def fake_extract_label_time_course(stc, labels, src, mode, verbose):
        events.append(("reduce", stc.data.shape[1]))
        return np.zeros((len(labels), stc.data.shape[1]))

    monkeypatch.setattr("mne.minimum_norm.apply_inverse_raw", fake_apply_inverse_raw)
    monkeypatch.setattr("mne.extract_label_time_course", fake_extract_label_time_course)

    result = processor._extract_raw_label_time_course(raw, inv=object())

    assert events == [
        ("apply", 0, 5),
        ("reduce", 5),
        ("apply", 5, 10),
        ("reduce", 5),
        ("apply", 10, 12),
        ("reduce", 2),
    ]
    assert result.shape == (1, 12)
    gc.collect()
    assert previous_stc() is None


def test_raw_chunk_releases_stc_when_reduction_fails(monkeypatch, memory_manager):
    processor = _processor(memory_manager, chunk_seconds=0.5)
    raw = DummyRaw()
    stc_ref = None

    class TrackedStc:
        data = np.zeros((2, 5))

    def fake_apply_inverse_raw(*args, **kwargs):
        nonlocal stc_ref
        stc = TrackedStc()
        stc_ref = weakref.ref(stc)
        return stc

    def fail_extract(*args, **kwargs):
        raise RuntimeError("injected reduction failure")

    monkeypatch.setattr("mne.minimum_norm.apply_inverse_raw", fake_apply_inverse_raw)
    monkeypatch.setattr("mne.extract_label_time_course", fail_extract)

    with pytest.raises(RuntimeError, match="injected reduction failure"):
        processor._extract_raw_label_time_course(raw, inv=object())

    gc.collect()
    assert stc_ref() is None
    memory_manager.cleanup.assert_called()


@pytest.mark.parametrize(
    "value",
    [None, 0, -1, math.nan, math.inf, -math.inf, 3600.1],
)
def test_chunk_seconds_requires_finite_positive_bounded_value(memory_manager, value):
    with pytest.raises((TypeError, ValueError)):
        SequentialProcessor(memory_manager=memory_manager, chunk_seconds=value)


def _attached_raw_with_annotations(*, orig_time):
    info = mne.create_info(["input"], 10.0, "eeg")
    if orig_time is not None:
        info.set_meas_date(orig_time)
    raw = mne.io.RawArray(
        np.zeros((1, 4)), info, first_samp=20, verbose=False
    )
    onset = 2.1 if orig_time is not None else 0.1
    raw.set_annotations(
        mne.Annotations(
            onset=[onset],
            duration=[0.2],
            description=["event"],
            orig_time=orig_time,
        )
    )
    return raw


@pytest.mark.parametrize(
    "orig_time",
    [None, datetime(2020, 1, 1, tzinfo=timezone.utc)],
)
def test_raw_output_preserves_attached_annotation_time_base(
    monkeypatch, memory_manager, tmp_path, orig_time
):
    processor = _processor(memory_manager)
    source = _attached_raw_with_annotations(orig_time=orig_time)
    monkeypatch.setattr(mne.io.BaseRaw, "export", MagicMock())

    raw, _ = processor._convert_raw_label_ts_to_eeg(
        np.zeros((1, 4)),
        sfreq=10.0,
        output_dir=str(tmp_path),
        subject_id="subject",
        first_samp=source.first_samp,
        meas_date=source.info["meas_date"],
        annotations=source.annotations,
    )

    assert raw.first_samp == source.first_samp
    assert raw.info["meas_date"] == source.info["meas_date"]
    assert raw.n_times == source.n_times == 4
    assert list(raw.annotations.description) == ["event"]
    np.testing.assert_allclose(raw.annotations.onset, source.annotations.onset)
    assert raw.annotations.orig_time == source.annotations.orig_time


def test_explicit_raw_chunking_capability_signal():
    from autoclean_eeg2source import RAW_CHUNKING_CAPABILITY

    assert RAW_CHUNKING_CAPABILITY == "sequential-raw-chunking-v1"


def test_epochs_processing_does_not_enter_raw_chunk_path(
    monkeypatch, memory_manager, tmp_path
):
    processor = _processor(memory_manager)
    epochs = MagicMock()
    epochs.ch_names = ["Cz"]
    epochs.info = {"sfreq": processor.resample_freq}
    processor.validator.validate_file_pair = MagicMock(
        return_value={"file_type": "epochs"}
    )
    processor.reader.read_epochs = MagicMock(return_value=epochs)
    processor._setup_fsaverage = MagicMock()
    processor._get_forward_solution = MagicMock(return_value=object())
    processor._convert_stc_to_eeg = MagicMock(
        return_value=(object(), str(tmp_path / "epochs.set"))
    )
    processor._convert_raw_inverse_to_eeg = MagicMock(
        side_effect=AssertionError("Epochs entered Raw chunk path")
    )
    monkeypatch.setattr(mne, "make_ad_hoc_cov", MagicMock(return_value=object()))
    monkeypatch.setattr(
        mne.minimum_norm, "make_inverse_operator", MagicMock(return_value=object())
    )
    apply_epochs = MagicMock(return_value=[SimpleNamespace()])
    monkeypatch.setattr(mne.minimum_norm, "apply_inverse_epochs", apply_epochs)

    result = processor.process_file("subject.set", str(tmp_path))

    assert result["status"] == "success"
    apply_epochs.assert_called_once()
    processor._convert_stc_to_eeg.assert_called_once()
    processor._convert_raw_inverse_to_eeg.assert_not_called()


def test_processor_family_defaults_to_one_worker(memory_manager, monkeypatch):
    monkeypatch.setattr(
        "autoclean_eeg2source.core.gpu_processor.check_gpu_availability",
        lambda: {"gpu_count": 0, "gpu_available": False},
    )

    parallel = ParallelProcessor(memory_manager=memory_manager, chunk_seconds=12.5)
    robust = RobustProcessor(memory_manager=memory_manager, chunk_seconds=8.0)
    gpu = GPUProcessor(memory_manager=memory_manager, chunk_seconds=6.0)

    assert parallel.n_jobs == 1
    assert gpu.n_jobs == 1
    assert robust.chunk_seconds == 8.0
    assert inspect.signature(run_standard_benchmark).parameters["n_jobs"].default == 1


def test_new_constructor_parameters_are_appended_after_legacy_surface(
    memory_manager, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "autoclean_eeg2source.core.gpu_processor.check_gpu_availability",
        lambda: {"gpu_count": 0, "gpu_available": False},
    )

    sequential = SequentialProcessor(memory_manager, "standard_1020", 128, 0.25)
    parallel = ParallelProcessor(
        memory_manager, "standard_1020", 128, 0.25, -1, 7, "threads"
    )
    cached = CachedProcessor(
        memory_manager,
        "standard_1020",
        128,
        0.25,
        -1,
        7,
        "threads",
        str(tmp_path),
    )
    gpu = GPUProcessor(
        memory_manager, "standard_1020", 128, 0.25, -1, 7, "threads", "none"
    )
    robust = RobustProcessor(
        memory_manager, "standard_1020", 128, 0.25, False, str(tmp_path)
    )

    assert sequential.chunk_seconds == 30.0
    assert parallel.n_jobs == mp.cpu_count()
    assert parallel.batch_size == 7
    assert parallel.parallel_method == "threads"
    assert cached.cache_dir == str(tmp_path)
    assert gpu.n_jobs == mp.cpu_count()
    assert gpu.batch_size == 7
    assert gpu.gpu_backend == "none"
    assert robust.recovery_mode is False
    assert robust.error_dir == str(tmp_path)
    assert all(
        processor.chunk_seconds == 30.0
        for processor in (parallel, cached, gpu, robust)
    )

    assert list(inspect.signature(ParallelProcessor).parameters)[-1] == "chunk_seconds"
    assert list(inspect.signature(CachedProcessor).parameters)[-1] == "chunk_seconds"
    assert list(inspect.signature(GPUProcessor).parameters)[-1] == "chunk_seconds"
    assert list(inspect.signature(RobustProcessor).parameters)[-1] == "chunk_seconds"
    assert list(inspect.signature(SequentialProcessor).parameters)[-1] == "chunk_seconds"


@pytest.mark.parametrize("processor_class", [ParallelProcessor, CachedProcessor, GPUProcessor])
def test_parallel_family_raw_routes_to_shared_bounded_implementation(
    monkeypatch, memory_manager, processor_class
):
    monkeypatch.setattr(
        "autoclean_eeg2source.core.gpu_processor.check_gpu_availability",
        lambda: {"gpu_count": 0, "gpu_available": False},
    )
    processor = processor_class(memory_manager=memory_manager)
    processor.validator.validate_file_pair = MagicMock(return_value={"file_type": "raw"})
    shared = MagicMock(return_value={"status": "success", "output_file": "raw.set"})
    monkeypatch.setattr(SequentialProcessor, "process_file", shared)

    result = processor.process_file("subject.set", "output")

    assert result["status"] == "success"
    shared.assert_called_once_with(processor, "subject.set", "output")


def test_explicit_legacy_all_core_workers_do_not_reach_forward_inverse(
    monkeypatch, memory_manager
):
    processor = ParallelProcessor(memory_manager=memory_manager, n_jobs=-1)
    processor.fsaverage_src = object()
    processor.fsaverage_bem = object()
    make_forward = MagicMock(return_value={"info": {"ch_names": []}})
    monkeypatch.setattr(mne, "make_forward_solution", make_forward)

    processor._get_forward_solution({"ch_names": ["Cz"]})

    assert processor.n_jobs == mp.cpu_count()
    assert make_forward.call_args.kwargs["n_jobs"] == 1


@pytest.mark.parametrize(
    "method_name",
    ["_apply_inverse_cupy", "_apply_inverse_pytorch", "_apply_inverse_tensorflow"],
)
def test_gpu_fallback_inverse_is_single_worker(
    monkeypatch, memory_manager, method_name
):
    monkeypatch.setattr(
        "autoclean_eeg2source.core.gpu_processor.check_gpu_availability",
        lambda: {"gpu_count": 0, "gpu_available": False},
    )
    processor = GPUProcessor(memory_manager=memory_manager, n_jobs=-1)
    apply_epochs = MagicMock(return_value=[])
    monkeypatch.setattr(mne.minimum_norm, "apply_inverse_epochs", apply_epochs)

    getattr(processor, method_name)(MagicMock(), object(), np.zeros((1, 1)))

    assert processor.n_jobs == mp.cpu_count()
    assert apply_epochs.call_args.kwargs["n_jobs"] == 1


def test_process_batch_resolves_legacy_all_core_sentinel(monkeypatch, memory_manager):
    captured = {}

    class FakeExecutor:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, *args, **kwargs):
            raise AssertionError("empty input must not submit work")

    monkeypatch.setattr(
        "autoclean_eeg2source.core.parallel_processor.ProcessPoolExecutor",
        FakeExecutor,
    )
    processor = ParallelProcessor(memory_manager=memory_manager, n_jobs=-1)

    assert processor.process_batch([], "unused", max_workers=-1) == []
    assert captured["max_workers"] == processor.n_jobs
    assert captured["max_workers"] > 0


def test_cli_batch_resolves_minus_one_without_individual_fallback(
    monkeypatch, tmp_path
):
    logger = MagicMock()
    processor = MagicMock()
    processor.n_jobs = 3

    def process_batch(files, output_dir, max_workers):
        assert files == ["subject.set"]
        assert max_workers == 3
        return [{"input_file": files[0], "status": "success"}]

    processor.process_batch.side_effect = process_batch
    monkeypatch.setattr(cli, "setup_logger", MagicMock(return_value=logger))
    monkeypatch.setattr(cli, "find_set_files", MagicMock(return_value=["subject.set"]))
    monkeypatch.setattr(cli, "MemoryManager", MagicMock())
    monkeypatch.setattr(cli, "ParallelProcessor", MagicMock(return_value=processor))
    individual = MagicMock()
    monkeypatch.setattr(cli, "_process_individual_file", individual)
    args = SimpleNamespace(
        log_level="INFO",
        log_file=None,
        error_dir=None,
        global_error_handler=False,
        input_path="input",
        recursive=False,
        output_dir=str(tmp_path),
        enable_cache=False,
        optimized_memory=False,
        max_memory=1.0,
        disk_offload=False,
        benchmark=False,
        robust=False,
        parallel=True,
        gpu=False,
        montage="standard_1020",
        resample_freq=None,
        lambda2=1.0 / 9.0,
        chunk_seconds=30.0,
        n_jobs=-1,
        batch_size=4,
        parallel_method="processes",
        gpu_backend="auto",
        batch_processing=True,
        save_summary=False,
    )

    assert cli.process_command(args) == 0
    processor.process_batch.assert_called_once()
    individual.assert_not_called()

"""Shared fixtures for tests."""

import os
import pytest
import numpy as np
import mne
from mne.epochs import EpochsArray


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_inverse: uses a real (non-mocked, network-fetched fsaverage) "
        "MNE inverse; opt-in only, set RUN_REAL_INVERSE_TESTS=1 to run "
        "(see tests/_real_inverse_fixtures.py)",
    )


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def create_clean_epochs():
    """Create clean synthetic test epochs."""
    # Create random data: 10 epochs, 32 channels, 100 timepoints
    data = np.random.randn(10, 32, 100) * 1e-6
    
    # Channel names for standard 10-20 system
    ch_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
        'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2',
        'AF3', 'AF4', 'F5', 'F1', 'F2', 'F6', 'FC5', 'FC1', 'FC2', 'FC6',
        'CP5', 'CP1'
    ]
    
    # Create info
    sfreq = 250.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * len(ch_names))
    
    # Create events
    events = np.array([[i, 0, 1] for i in range(10)])
    
    # Create epochs
    epochs = EpochsArray(data, info, events, tmin=0)
    
    return epochs


@pytest.fixture
def create_set_file(temp_dir, create_clean_epochs):
    """Create a synthetic .set file."""
    try:
        # Try to actually save a real EEGLAB .set file
        set_file = os.path.join(temp_dir, "test_epochs.set")
        create_clean_epochs.export(set_file, fmt='eeglab', overwrite=True)
        
        # If we get here, export worked
        return set_file
        
    except Exception:
        # Fall back to creating a dummy file
        set_file = os.path.join(temp_dir, "test_epochs.set")
        with open(set_file, "w") as f:
            f.write("DUMMY EEGLAB SET FILE")
        
        # Pretend to create the .fdt file too
        fdt_file = os.path.join(temp_dir, "test_epochs.fdt")
        with open(fdt_file, "wb") as f:
            # Write some binary data
            data = np.random.randn(10, 32, 100).astype(np.float32)
            f.write(data.tobytes())
        
        return set_file


@pytest.fixture
def mock_mne_functions(monkeypatch, create_clean_epochs):
    """Mock key MNE functions for testing."""
    
    # Mock io.read_epochs_eeglab
    def mock_read_epochs(*args, **kwargs):
        return create_clean_epochs
    
    # Mock source_spaces
    mock_src = MagicMock()
    
    # Mock forward solution
    mock_fwd = MagicMock()
    
    # Mock inverse operator
    mock_inv = MagicMock()
    
    # Mock source estimates
    mock_stcs = [MagicMock() for _ in range(10)]
    for i, stc in enumerate(mock_stcs):
        stc.data = np.random.randn(68, 100) * 1e-8
        stc.tmin = 0.0
        stc.tstep = 1.0 / 250
    
    # Apply mocks
    monkeypatch.setattr(mne.io, 'read_epochs_eeglab', mock_read_epochs)
    monkeypatch.setattr(mne, 'read_source_spaces', lambda *args, **kwargs: mock_src)
    monkeypatch.setattr(mne, 'make_forward_solution', lambda *args, **kwargs: mock_fwd)
    monkeypatch.setattr(mne.minimum_norm, 'make_inverse_operator', lambda *args, **kwargs: mock_inv)
    monkeypatch.setattr(mne.minimum_norm, 'apply_inverse_epochs', lambda *args, **kwargs: mock_stcs)
    
    class MockLabels:
        def __init__(self, n=68):
            self.labels = []
            for i in range(n):
                label = MagicMock()
                label.name = f"region-{i}"
                hemisphere = "lh" if i < n // 2 else "rh"
                label.name = f"{label.name}-{hemisphere}"
                label.pos = np.random.randn(10, 3) * 0.1
                self.labels.append(label)
        
        def __iter__(self):
            return iter(self.labels)
        
        def __len__(self):
            return len(self.labels)
    
    monkeypatch.setattr(mne, 'read_labels_from_annot', lambda *args, **kwargs: MockLabels().labels)
    
    return {
        'epochs': create_clean_epochs,
        'src': mock_src,
        'fwd': mock_fwd,
        'inv': mock_inv,
        'stcs': mock_stcs
    }
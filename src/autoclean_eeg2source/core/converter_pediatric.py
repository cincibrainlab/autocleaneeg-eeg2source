"""EEG to source conversion with pediatric/infant head model support.

This is a DRAFT version that extends the standard converter to support
MNE's infant templates (2 weeks to 2 years) in addition to fsaverage.

Available infant ages: '2wk', '1mo', '2mo', '3mo', '4.5mo', '6mo',
                       '7.5mo', '9mo', '10.5mo', '12mo', '15mo', '18mo', '2yr'
"""

import os
import gc
import logging
from typing import Optional, Dict, Any, Literal

import numpy as np
import mne
from mne.datasets import fetch_fsaverage, fetch_infant_template
import pandas as pd

from ..io.eeglab_reader import EEGLABReader
from ..io.validators import EEGLABValidator
from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)

# Valid infant template ages
INFANT_AGES = {
    '2wk', '1mo', '2mo', '3mo', '4.5mo', '6mo',
    '7.5mo', '9mo', '10.5mo', '12mo', '15mo', '18mo', '2yr'
}


class PediatricProcessor:
    """Processor for EEG to source localization with pediatric head model support.

    This processor supports both the standard fsaverage adult template and
    MNE's infant templates for ages 2 weeks to 2 years.

    Parameters
    ----------
    head_model : str
        Head model to use. Options:
        - 'fsaverage': Standard adult template (default)
        - Infant ages: '2wk', '1mo', '2mo', '3mo', '4.5mo', '6mo',
                       '7.5mo', '9mo', '10.5mo', '12mo', '15mo', '18mo', '2yr'
    memory_manager : MemoryManager, optional
        Memory manager instance
    montage : str
        EEG montage name
    resample_freq : float
        Target sampling frequency
    lambda2 : float
        Regularization parameter for inverse solution
    subjects_dir : str, optional
        Custom subjects directory for MRI templates. If None, uses MNE default.

    Examples
    --------
    # Adult processing (default)
    >>> processor = PediatricProcessor(head_model='fsaverage')

    # 6-month infant processing
    >>> processor = PediatricProcessor(head_model='6mo')

    # 2-year-old toddler processing
    >>> processor = PediatricProcessor(head_model='2yr')
    """

    def __init__(self,
                 head_model: str = "fsaverage",
                 memory_manager: Optional[MemoryManager] = None,
                 montage: str = "GSN-HydroCel-129",
                 resample_freq: float = 250,
                 lambda2: float = 1.0 / 9.0,
                 subjects_dir: Optional[str] = None):
        """Initialize pediatric processor."""

        # Validate head model
        if head_model != 'fsaverage' and head_model not in INFANT_AGES:
            raise ValueError(
                f"Invalid head_model '{head_model}'. Must be 'fsaverage' or one of: "
                f"{sorted(INFANT_AGES)}"
            )

        self.head_model = head_model
        self.is_infant = head_model in INFANT_AGES
        self.memory_manager = memory_manager or MemoryManager()
        self.montage = montage
        self.resample_freq = resample_freq
        self.lambda2 = lambda2
        self.subjects_dir = subjects_dir

        # Cache for forward solution and source space
        self.forward_solution = None
        self.src = None
        self.bem = None
        self.labels = None
        self.subject_name = None  # Will be set during setup

        # Initialize components
        self.reader = EEGLABReader(memory_manager=self.memory_manager)
        self.validator = EEGLABValidator()

        model_type = "infant" if self.is_infant else "adult"
        logger.info(
            f"Initialized PediatricProcessor: head_model={head_model} ({model_type}), "
            f"montage={montage}, resample={resample_freq}Hz"
        )

    def _setup_head_model(self):
        """Setup the head model (fsaverage or infant template)."""
        if self.src is not None:
            return  # Already setup

        if self.is_infant:
            self._setup_infant_template()
        else:
            self._setup_fsaverage()

    def _setup_fsaverage(self):
        """Setup fsaverage adult brain model."""
        logger.info("Setting up fsaverage (adult) brain model...")

        # Fetch fsaverage files
        fs_dir = fetch_fsaverage(verbose=False)
        self.subjects_dir = self.subjects_dir or os.path.dirname(fs_dir)
        self.subject_name = 'fsaverage'

        # Load source space (ico-5)
        self.src = mne.read_source_spaces(
            os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
        )

        # Load BEM solution
        self.bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

        # Load labels for DK atlas
        self.labels = mne.read_labels_from_annot(
            'fsaverage', parc='aparc', subjects_dir=self.subjects_dir
        )
        self.labels = [label for label in self.labels if 'unknown' not in label.name]

        logger.info(f"Loaded {len(self.labels)} brain regions from DK atlas (fsaverage)")

    def _setup_infant_template(self):
        """Setup infant brain model template.

        Note: Infant templates have different file structures than fsaverage.
        The exact paths may need adjustment based on MNE version.
        """
        logger.info(f"Setting up infant brain model for age: {self.head_model}...")

        # Fetch infant template - this downloads if needed
        # The function returns the path to the subject directory
        if self.subjects_dir is None:
            # Use default MNE subjects directory
            self.subjects_dir = mne.get_config('SUBJECTS_DIR') or os.path.join(
                mne.get_config('MNE_DATA', os.path.expanduser('~/mne_data')),
                'MNE-infant-data'
            )

        # Fetch the infant template
        self.subject_name = fetch_infant_template(
            age=self.head_model,
            subjects_dir=self.subjects_dir,
            verbose=True
        )

        logger.info(f"Infant template subject: {self.subject_name}")

        subject_path = os.path.join(self.subjects_dir, self.subject_name)
        bem_path = os.path.join(subject_path, 'bem')

        # List available files in the bem directory
        if os.path.exists(bem_path):
            bem_files = os.listdir(bem_path)
            logger.info(f"Available BEM files: {bem_files}")

        # Try to find source space file
        # Infant templates typically use ico-4 or ico-5
        src_candidates = [
            f"{self.subject_name}-ico-5-src.fif",
            f"{self.subject_name}-ico-4-src.fif",
            f"ANTS{self.head_model}-ico-5-src.fif",
            f"ANTS{self.head_model}-ico-4-src.fif",
        ]

        src_file = None
        for candidate in src_candidates:
            full_path = os.path.join(bem_path, candidate)
            if os.path.exists(full_path):
                src_file = full_path
                break

        if src_file is None:
            # Need to compute source space
            logger.warning(
                f"No pre-computed source space found for {self.head_model}. "
                "Computing source space (this may take a few minutes)..."
            )
            self.src = mne.setup_source_space(
                self.subject_name,
                spacing='ico4',  # Use ico-4 for faster computation
                subjects_dir=self.subjects_dir,
                add_dist=False,
                verbose=True
            )
            # Save for future use
            src_file = os.path.join(bem_path, f"{self.subject_name}-ico-4-src.fif")
            mne.write_source_spaces(src_file, self.src, overwrite=True)
            logger.info(f"Saved source space to {src_file}")
        else:
            self.src = mne.read_source_spaces(src_file)
            logger.info(f"Loaded source space from {src_file}")

        # Find BEM solution
        bem_candidates = [
            f"{self.subject_name}-5120-5120-5120-bem-sol.fif",
            f"{self.subject_name}-bem-sol.fif",
            f"ANTS{self.head_model}-5120-5120-5120-bem-sol.fif",
        ]

        self.bem = None
        for candidate in bem_candidates:
            full_path = os.path.join(bem_path, candidate)
            if os.path.exists(full_path):
                self.bem = full_path
                break

        if self.bem is None:
            # Need to compute BEM model
            logger.warning(
                f"No pre-computed BEM solution found for {self.head_model}. "
                "Computing BEM model (this may take several minutes)..."
            )
            # Create BEM model
            model = mne.make_bem_model(
                self.subject_name,
                subjects_dir=self.subjects_dir,
                conductivity=(0.3, 0.006, 0.3),  # Standard 3-layer
                verbose=True
            )
            self.bem = mne.make_bem_solution(model)
            # Note: BEM solution object, not file path
            logger.info("BEM solution computed successfully")
        else:
            logger.info(f"Using BEM solution: {self.bem}")

        # Load labels - infant templates may have different parcellations
        try:
            self.labels = mne.read_labels_from_annot(
                self.subject_name, parc='aparc', subjects_dir=self.subjects_dir
            )
            self.labels = [label for label in self.labels if 'unknown' not in label.name]
            logger.info(f"Loaded {len(self.labels)} brain regions from DK atlas")
        except Exception as e:
            logger.warning(f"Could not load DK atlas labels: {e}")
            logger.warning("Attempting to use generic parcellation...")
            # Try other parcellations
            for parc in ['aparc', 'aparc.a2009s', 'PALS_B12_Brodmann']:
                try:
                    self.labels = mne.read_labels_from_annot(
                        self.subject_name, parc=parc, subjects_dir=self.subjects_dir
                    )
                    self.labels = [l for l in self.labels if 'unknown' not in l.name]
                    logger.info(f"Loaded {len(self.labels)} regions using {parc} parcellation")
                    break
                except:
                    continue

            if self.labels is None:
                raise RuntimeError(
                    f"Could not load any parcellation for {self.subject_name}. "
                    "You may need to run FreeSurfer parcellation on the template."
                )

    def _get_forward_solution(self, info: mne.Info) -> mne.Forward:
        """Get cached or compute forward solution."""
        if self.forward_solution is not None:
            # Check if montage matches
            if len(self.forward_solution['info']['ch_names']) == len(info['ch_names']):
                logger.debug("Using cached forward solution")
                return self.forward_solution

        logger.info(f"Computing forward solution for {self.head_model}...")
        self.memory_manager.check_available()

        # Compute forward solution
        self.forward_solution = mne.make_forward_solution(
            info,
            trans=self.subject_name,  # Use subject name for trans
            src=self.src,
            bem=self.bem,
            eeg=True,
            mindist=5.0,
            n_jobs=1
        )

        self.memory_manager.log_memory_status("After forward solution")
        return self.forward_solution

    def process_file(self, input_file: str, output_dir: str) -> Dict[str, Any]:
        """
        Process single EEG file with memory monitoring.

        Parameters
        ----------
        input_file : str
            Path to input .set file
        output_dir : str
            Output directory

        Returns
        -------
        result : dict
            Processing result with status and output file
        """
        result = {
            'input_file': input_file,
            'head_model': self.head_model,
            'status': 'failed',
            'output_file': None,
            'error': None
        }

        try:
            logger.info(f"Processing: {os.path.basename(input_file)} with {self.head_model} head model")
            report = self.validator.validate_file_pair(input_file)

            # Check memory before starting
            self.memory_manager.check_available()

            # Setup head model (fsaverage or infant)
            self._setup_head_model()

            # Load epochs
            if report['file_type'] == 'epochs':
                epochs = self.reader.read_epochs(input_file)
            else:
                epochs = self.reader.read_raw(input_file)

            # Pick EEG channels
            logger.info("Selecting EEG channels only")
            eog_channels = [ch for ch in epochs.ch_names
                          if any(eog in ch.upper() for eog in ['EOG', 'HEOG', 'VEOG'])]
            if eog_channels:
                logger.info(f"Setting {len(eog_channels)} EOG channels: {eog_channels}")
                epochs.set_channel_types({ch: 'eog' for ch in eog_channels})

            epochs.pick("eeg")

            # Set montage
            logger.info(f"Setting montage: {self.montage}")
            epochs.set_montage(
                mne.channels.make_standard_montage(self.montage),
                match_case=False
            )

            # Resample if needed
            if epochs.info['sfreq'] != self.resample_freq:
                logger.info(f"Resampling from {epochs.info['sfreq']}Hz to {self.resample_freq}Hz")
                epochs.resample(self.resample_freq)

            # Set EEG reference
            epochs.set_eeg_reference(projection=True)

            # Get forward solution
            fwd = self._get_forward_solution(epochs.info)

            # Compute noise covariance
            logger.info("Computing noise covariance...")
            noise_cov = mne.make_ad_hoc_cov(epochs.info)

            # Create inverse operator
            logger.info("Creating inverse operator...")
            inv = mne.minimum_norm.make_inverse_operator(
                epochs.info, fwd, noise_cov, verbose=False
            )

            # Apply inverse solution
            logger.info("Applying inverse solution to epochs...")
            if report['file_type'] == 'epochs':
                stcs = mne.minimum_norm.apply_inverse_epochs(
                    epochs, inv, lambda2=self.lambda2, method="MNE",
                    pick_ori='normal', verbose=False
                )
            else:
                stcs = mne.minimum_norm.apply_inverse_raw(
                    epochs, inv, lambda2=self.lambda2, method="MNE",
                    pick_ori='normal', verbose=False
                )

            # Convert to EEG format
            logger.info("Converting source estimates to EEG format...")
            if report['file_type'] == 'epochs':
                output_epochs, output_file = self._convert_stc_to_eeg(
                    stcs, output_dir,
                    subject_id=os.path.splitext(os.path.basename(input_file))[0],
                    original_epochs=epochs
                )
            else:
                output_epochs, output_file = self._convert_raw_stc_to_eeg(
                    stcs, output_dir,
                    subject_id=os.path.splitext(os.path.basename(input_file))[0]
                )

            result['status'] = 'success'
            result['output_file'] = output_file

            # Cleanup
            del epochs, inv, stcs, output_epochs
            gc.collect()
            self.memory_manager.cleanup()

            logger.info(f"✓ Successfully processed: {output_file}")

        except MemoryError as e:
            logger.error(f"Memory exhausted: {e}")
            result['error'] = str(e)
            gc.collect()
            self.memory_manager.cleanup()

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            result['error'] = str(e)
            import traceback
            logger.debug(traceback.format_exc())

        return result

    def _convert_stc_to_eeg(self, stc_list: list, output_dir: str,
                           subject_id: str, original_epochs=None) -> tuple:
        """Convert source estimates to EEG format with brain region labels."""
        logger.info(f"Converting {len(stc_list)} source estimates to EEG format...")

        # Extract time series for each label
        all_label_ts = []
        for stc in stc_list:
            label_ts = mne.extract_label_time_course(
                stc, self.labels, src=self.src,
                mode='mean', verbose=False
            )
            all_label_ts.append(label_ts)

        # Stack to 3D array
        label_data = np.array(all_label_ts)

        n_epochs = len(stc_list)
        n_regions = len(self.labels)
        sfreq = 1.0 / stc_list[0].tstep
        ch_names = [label.name for label in self.labels]

        # Create channel positions
        ch_pos = {}
        for i, label in enumerate(self.labels):
            if hasattr(label, 'pos') and len(label.pos) > 0:
                centroid = np.mean(label.pos, axis=0)
            else:
                phi = (1 + np.sqrt(5)) / 2
                idx = i + 1
                theta = 2 * np.pi * idx / phi**2
                phi_angle = np.arccos(1 - 2 * ((idx % phi**2) / phi**2))
                centroid = np.array([
                    np.sin(phi_angle) * np.cos(theta),
                    np.sin(phi_angle) * np.sin(theta),
                    np.cos(phi_angle)
                ]) * 0.1
            ch_pos[label.name] = centroid

        # Create MNE Info
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=['eeg'] * n_regions
        )

        for idx, ch_name in enumerate(ch_names):
            info['chs'][idx]['loc'][:3] = ch_pos[ch_name]

        # Create events
        if original_epochs is not None and hasattr(original_epochs, 'event_id'):
            event_id = original_epochs.event_id.copy()
            epoch_duration = stc_list[0].times[-1] - stc_list[0].times[0]
            epoch_length_samples = int(sfreq * epoch_duration)
            padding_samples = int(sfreq * 0.1)
            epoch_spacing = epoch_length_samples + padding_samples

            if len(original_epochs.events) == n_epochs:
                events = np.array([
                    [i * epoch_spacing, 0, original_epochs.events[i, 2]]
                    for i in range(n_epochs)
                ])
            else:
                first_event_code = list(event_id.values())[0]
                events = np.array([
                    [i * epoch_spacing, 0, first_event_code]
                    for i in range(n_epochs)
                ])
        else:
            epoch_duration = stc_list[0].times[-1] - stc_list[0].times[0]
            epoch_length_samples = int(sfreq * epoch_duration)
            padding_samples = int(sfreq * 0.1)
            epoch_spacing = epoch_length_samples + padding_samples
            events = np.array([
                [i * epoch_spacing, 0, 1] for i in range(n_epochs)
            ])
            event_id = {'event': 1}

        tmin = stc_list[0].tmin

        epochs = mne.EpochsArray(
            label_data, info, events=events,
            event_id=event_id, tmin=tmin
        )

        # Save output
        os.makedirs(output_dir, exist_ok=True)

        # Include head model in filename
        model_suffix = f"_{self.head_model}" if self.is_infant else ""
        output_file = os.path.join(output_dir, f"{subject_id}{model_suffix}_dk_regions.set")
        epochs.export(output_file, fmt='eeglab', overwrite=True)

        logger.info(f"Saved {n_regions} regions to {output_file}")

        # Save metadata
        self._save_metadata(output_dir, subject_id, ch_names, ch_pos)

        return epochs, output_file

    def _convert_raw_stc_to_eeg(self, stc, output_dir: str, subject_id: str) -> tuple:
        """Convert raw source estimate to EEG format."""
        logger.info("Converting raw source estimate to EEG format...")

        label_ts = mne.extract_label_time_course(
            stc, self.labels, src=self.src,
            mode='mean', verbose=False
        )

        n_regions = len(self.labels)
        sfreq = 1.0 / stc.tstep
        ch_names = [label.name for label in self.labels]

        ch_pos = {}
        for i, label in enumerate(self.labels):
            if hasattr(label, 'pos') and len(label.pos) > 0:
                centroid = np.mean(label.pos, axis=0)
            else:
                phi = (1 + np.sqrt(5)) / 2
                idx = i + 1
                theta = 2 * np.pi * idx / phi**2
                phi_angle = np.arccos(1 - 2 * ((idx % phi**2) / phi**2))
                centroid = np.array([
                    np.sin(phi_angle) * np.cos(theta),
                    np.sin(phi_angle) * np.sin(theta),
                    np.cos(phi_angle)
                ]) * 0.1
            ch_pos[label.name] = centroid

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=['eeg'] * n_regions
        )

        for idx, ch_name in enumerate(ch_names):
            info['chs'][idx]['loc'][:3] = ch_pos[ch_name]

        raw = mne.io.RawArray(label_ts, info, first_samp=0, verbose=False)

        os.makedirs(output_dir, exist_ok=True)
        model_suffix = f"_{self.head_model}" if self.is_infant else ""
        output_file = os.path.join(output_dir, f"{subject_id}{model_suffix}_dk_regions.set")
        raw.export(output_file, fmt='eeglab', overwrite=True)

        logger.info(f"Saved {n_regions} regions to {output_file}")
        self._save_metadata(output_dir, subject_id, ch_names, ch_pos)

        return raw, output_file

    def _save_metadata(self, output_dir: str, subject_id: str,
                      ch_names: list, ch_pos: dict):
        """Save region metadata to CSV file."""
        region_info = {
            'names': ch_names,
            'hemisphere': ['lh' if '-lh' in name else 'rh' for name in ch_names],
            'x': [ch_pos[name][0] for name in ch_names],
            'y': [ch_pos[name][1] for name in ch_names],
            'z': [ch_pos[name][2] for name in ch_names],
            'head_model': [self.head_model] * len(ch_names)
        }

        model_suffix = f"_{self.head_model}" if self.is_infant else ""
        info_file = os.path.join(output_dir, f"{subject_id}{model_suffix}_region_info.csv")
        pd.DataFrame(region_info).to_csv(info_file, index=False)
        logger.debug(f"Saved region info to {info_file}")


def get_available_head_models() -> dict:
    """Return available head models with descriptions.

    Returns
    -------
    dict
        Dictionary mapping model names to descriptions
    """
    models = {
        'fsaverage': 'Adult template (FreeSurfer average brain)',
    }

    infant_descriptions = {
        '2wk': '2 weeks old',
        '1mo': '1 month old',
        '2mo': '2 months old',
        '3mo': '3 months old',
        '4.5mo': '4.5 months old',
        '6mo': '6 months old',
        '7.5mo': '7.5 months old',
        '9mo': '9 months old',
        '10.5mo': '10.5 months old',
        '12mo': '12 months old (1 year)',
        '15mo': '15 months old',
        '18mo': '18 months old',
        '2yr': '2 years old',
    }

    models.update(infant_descriptions)
    return models

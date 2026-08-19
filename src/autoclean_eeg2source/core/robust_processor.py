"""Robust processor with error recovery for EEG to source conversion."""

import os
import gc
import json
import logging
import traceback
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import mne
import pandas as pd
from datetime import datetime

from .converter import SequentialProcessor
from .memory_manager import MemoryManager
from ..io.validators import EEGLABValidator
from ..io.data_quality import QualityAssessor
from ..io.exceptions import (
    EEGLABError, FileFormatError, FileMismatchError, 
    ChannelError, MontageError, CorruptedDataError, 
    DataQualityError, ProcessingError
)

logger = logging.getLogger(__name__)


class RobustProcessor(SequentialProcessor):
    """Robust processor with error recovery for EEG to source conversion."""
    
    def __init__(self, 
                 memory_manager: Optional[MemoryManager] = None,
                 montage: str = "GSN-HydroCel-129",
                 resample_freq: float = 250,
                 lambda2: float = 1.0 / 9.0,
                 recovery_mode: bool = True,
                 error_dir: Optional[str] = None,
                 chunk_seconds: float = 30.0):
        """
        Initialize robust processor.
        
        Parameters
        ----------
        memory_manager : MemoryManager, optional
            Memory manager instance
        montage : str
            EEG montage name
        resample_freq : float
            Target sampling frequency
        lambda2 : float
            Regularization parameter for inverse solution
        recovery_mode : bool
            Whether to use recovery strategies for errors
        error_dir : str, optional
            Directory to save error reports
        chunk_seconds : float, optional
            Continuous/raw inverse chunk length in seconds
        """
        super().__init__(
            memory_manager=memory_manager,
            montage=montage,
            resample_freq=resample_freq,
            lambda2=lambda2,
            chunk_seconds=chunk_seconds
        )
        
        self.recovery_mode = recovery_mode
        self.error_dir = error_dir
        
        # Enhanced validators
        self.validator = EEGLABValidator()
        self.quality_assessor = QualityAssessor()
        
        # Recovery statistics
        self.recovery_stats = {
            'attempted': 0,
            'successful': 0,
            'failed': 0,
            'strategies_used': {}
        }
        
        logger.info(
            f"Initialized robust processor with recovery_mode={recovery_mode}"
        )
    
    def process_with_recovery(self, input_file: str, output_dir: str) -> Dict[str, Any]:
        """
        Process file with error recovery strategies.
        
        Parameters
        ----------
        input_file : str
            Path to input .set file
        output_dir : str
            Output directory
            
        Returns
        -------
        result : dict
            Processing result with detailed status
        """
        result = {
            'input_file': input_file,
            'status': 'failed',
            'output_file': None,
            'error': None,
            'recovery_attempted': False,
            'recovery_successful': False,
            'recovery_strategy': None,
            'warnings': [],
            'processing_time': None
        }
        
        start_time = datetime.now()
        
        # Try normal processing first
        try:
            logger.info(f"Attempting normal processing for {os.path.basename(input_file)}")
            
            # Validate with enhanced validator
            validation = self.validator.check_all(
                input_file, montage_name=self.montage
            )
            
            if not validation['valid']:
                if 'file_validation' in validation and not validation['file_validation']['valid']:
                    error_msg = "; ".join(validation['file_validation'].get('errors', []))
                    raise FileFormatError(f"File validation failed: {error_msg}")
                elif 'montage_validation' in validation and not validation['montage_validation']['valid']:
                    error_msg = "; ".join(validation['montage_validation'].get('errors', []))
                    raise MontageError(f"Montage validation failed: {error_msg}")
                elif 'quality_validation' in validation and validation['quality_validation'].get('issues_found', False):
                    raise DataQualityError("Data quality issues found")
                else:
                    raise ValueError("Validation failed for unknown reason")
            
            # If validation passes, proceed with standard processing
            normal_result = super().process_file(input_file, output_dir)
            
            # If successful, return result
            if normal_result['status'] == 'success':
                result.update(normal_result)
                result['warnings'] = validation.get('file_validation', {}).get('warnings', [])
                result['processing_time'] = (datetime.now() - start_time).total_seconds()
                return result
                
            # If standard processing failed, try recovery
            raise ProcessingError(normal_result.get('error', "Unknown processing error"))
            
        except Exception as e:
            # If not in recovery mode, just return error
            if not self.recovery_mode:
                result['error'] = str(e)
                result['exception_type'] = type(e).__name__
                result['processing_time'] = (datetime.now() - start_time).total_seconds()
                
                # Save error report
                self._save_error_report(input_file, e, result)
                return result
            
            # Otherwise attempt recovery
            logger.warning(f"Normal processing failed: {str(e)}")
            return self._attempt_recovery(input_file, output_dir, e, result, start_time)
    
    def _attempt_recovery(self, input_file: str, output_dir: str, 
                          exception: Exception, result: Dict[str, Any],
                          start_time: datetime) -> Dict[str, Any]:
        """Attempt recovery strategies based on the error type."""
        self.recovery_stats['attempted'] += 1
        result['recovery_attempted'] = True
        
        # Choose recovery strategy based on exception type
        try:
            if isinstance(exception, (FileFormatError, FileMismatchError)):
                return self._recover_file_issues(input_file, output_dir, exception, result, start_time)
            elif isinstance(exception, MontageError):
                return self._recover_montage_issues(input_file, output_dir, exception, result, start_time)
            elif isinstance(exception, (DataQualityError, CorruptedDataError)):
                return self._recover_data_quality(input_file, output_dir, exception, result, start_time)
            elif isinstance(exception, MemoryError):
                return self._recover_memory_issues(input_file, output_dir, exception, result, start_time)
            else:
                # Generic fallback recovery
                return self._recover_generic(input_file, output_dir, exception, result, start_time)
        except Exception as recovery_e:
            # Recovery itself failed
            self.recovery_stats['failed'] += 1
            
            result['status'] = 'failed'
            result['error'] = f"Recovery failed: {str(recovery_e)}"
            result['original_error'] = str(exception)
            result['recovery_successful'] = False
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            # Save error report
            self._save_error_report(input_file, recovery_e, result, original_error=exception)
            
            return result
    
    def _recover_file_issues(self, input_file: str, output_dir: str, 
                            exception: Exception, result: Dict[str, Any],
                            start_time: datetime) -> Dict[str, Any]:
        """Recovery strategy for file format issues."""
        strategy = "file_recovery"
        self._count_strategy(strategy)
        result['recovery_strategy'] = strategy
        
        logger.info(f"Attempting file issue recovery for {os.path.basename(input_file)}")
        
        try:
            # Try reading with different options
            import mne
            
            # Strategy 1: Try direct read without validation
            logger.info("Recovery strategy: Direct epoch reading")
            try:
                epochs = mne.io.read_epochs_eeglab(input_file, verbose=False)
                
                # If we got here, we could read the file directly
                # Proceed with modified processing
                modified_result = self._process_with_epochs(
                    epochs, output_dir, 
                    subject_id=os.path.splitext(os.path.basename(input_file))[0]
                )
                
                if modified_result['status'] == 'success':
                    self.recovery_stats['successful'] += 1
                    result.update(modified_result)
                    result['recovery_successful'] = True
                    result['warnings'].append("Recovery used direct epoch reading")
                    result['processing_time'] = (datetime.now() - start_time).total_seconds()
                    return result
            except Exception as e:
                logger.warning(f"Direct reading failed: {e}")
            
            # Strategy failed
            raise ProcessingError(f"File recovery strategies exhausted: {str(exception)}")
            
        except Exception as e:
            # Recovery failed
            self.recovery_stats['failed'] += 1
            
            result['status'] = 'failed'
            result['error'] = f"File recovery failed: {str(e)}"
            result['original_error'] = str(exception)
            result['recovery_successful'] = False
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            # Save error report
            self._save_error_report(input_file, e, result, original_error=exception)
            
            return result
    
    def _recover_montage_issues(self, input_file: str, output_dir: str, 
                               exception: Exception, result: Dict[str, Any],
                               start_time: datetime) -> Dict[str, Any]:
        """Recovery strategy for montage issues."""
        strategy = "montage_recovery"
        self._count_strategy(strategy)
        result['recovery_strategy'] = strategy
        
        logger.info(f"Attempting montage issue recovery for {os.path.basename(input_file)}")
        
        try:
            # Try with different montages
            import mne
            
            # Read the epochs
            epochs = mne.io.read_epochs_eeglab(input_file, verbose=False)
            
            # Try standard montages that might match the channel count
            n_channels = len(epochs.ch_names)
            
            # Strategy 1: Try to find a montage with similar channel count
            alternative_montages = [
                "standard_1005",   # 345 channels
                "standard_1020",   # 97 channels
                "GSN-HydroCel-256", # 256 channels
                "GSN-HydroCel-129", # 129 channels
                "GSN-HydroCel-128", # 128 channels
                "GSN-HydroCel-64",  # 64 channels
                "GSN-HydroCel-32",  # 32 channels
                "easycap-M1",       # 74 channels
                "biosemi64",        # 64 channels
                "biosemi32",        # 32 channels
                "biosemi16",        # 16 channels
            ]
            
            for alt_montage in alternative_montages:
                try:
                    montage = mne.channels.make_standard_montage(alt_montage)
                    
                    # If channels are within 5% of each other, try this montage
                    montage_ch_count = len(montage.ch_names)
                    if abs(n_channels - montage_ch_count) <= max(n_channels * 0.05, 2):
                        logger.info(f"Trying alternative montage: {alt_montage}")
                        
                        # Copy with new montage settings
                        processor = SequentialProcessor(
                            memory_manager=self.memory_manager,
                            montage=alt_montage,
                            resample_freq=self.resample_freq,
                            lambda2=self.lambda2,
                            chunk_seconds=self.chunk_seconds
                        )
                        
                        # Try processing
                        modified_result = processor.process_file(input_file, output_dir)
                        
                        if modified_result['status'] == 'success':
                            self.recovery_stats['successful'] += 1
                            result.update(modified_result)
                            result['recovery_successful'] = True
                            result['warnings'].append(
                                f"Recovery used alternative montage: {alt_montage}"
                            )
                            result['processing_time'] = (datetime.now() - start_time).total_seconds()
                            return result
                except Exception as e:
                    logger.warning(f"Alternative montage {alt_montage} failed: {e}")
            
            # Strategy 2: Try to ignore montage
            try:
                logger.info("Recovery strategy: Skipping montage assignment")
                
                # Create a dummy montage
                ch_pos = {}
                for i, name in enumerate(epochs.ch_names):
                    angle = i * 2 * np.pi / len(epochs.ch_names)
                    ch_pos[name] = np.array([
                        0.9 * np.cos(angle),
                        0.9 * np.sin(angle),
                        0
                    ])
                
                # Set a synthetic montage
                montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
                epochs.set_montage(montage)
                
                # Process with epochs that have a dummy montage
                modified_result = self._process_with_epochs(
                    epochs, output_dir, 
                    subject_id=os.path.splitext(os.path.basename(input_file))[0]
                )
                
                if modified_result['status'] == 'success':
                    self.recovery_stats['successful'] += 1
                    result.update(modified_result)
                    result['recovery_successful'] = True
                    result['warnings'].append("Recovery used synthetic montage")
                    result['processing_time'] = (datetime.now() - start_time).total_seconds()
                    return result
            except Exception as e:
                logger.warning(f"Synthetic montage failed: {e}")
            
            # Strategy failed
            raise ProcessingError(f"Montage recovery strategies exhausted: {str(exception)}")
            
        except Exception as e:
            # Recovery failed
            self.recovery_stats['failed'] += 1
            
            result['status'] = 'failed'
            result['error'] = f"Montage recovery failed: {str(e)}"
            result['original_error'] = str(exception)
            result['recovery_successful'] = False
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            # Save error report
            self._save_error_report(input_file, e, result, original_error=exception)
            
            return result
    
    def _recover_data_quality(self, input_file: str, output_dir: str, 
                             exception: Exception, result: Dict[str, Any],
                             start_time: datetime) -> Dict[str, Any]:
        """Recovery strategy for data quality issues."""
        strategy = "quality_recovery"
        self._count_strategy(strategy)
        result['recovery_strategy'] = strategy
        
        logger.info(f"Attempting data quality recovery for {os.path.basename(input_file)}")
        
        try:
            # Read the epochs
            import mne
            epochs = mne.io.read_epochs_eeglab(input_file, verbose=False)
            
            # Strategy 1: Fix data quality issues
            logger.info("Recovery strategy: Quality assessment and fixing")
            fixed_epochs, quality_report = self.quality_assessor.fix_epochs(epochs)
            
            # If fixing succeeded, process with fixed epochs
            if 'actions' in quality_report and quality_report['actions']:
                logger.info(f"Applied quality fixes: {quality_report['actions']}")
                
                # Process with fixed epochs
                modified_result = self._process_with_epochs(
                    fixed_epochs, output_dir, 
                    subject_id=os.path.splitext(os.path.basename(input_file))[0]
                )
                
                if modified_result['status'] == 'success':
                    self.recovery_stats['successful'] += 1
                    result.update(modified_result)
                    result['recovery_successful'] = True
                    result['warnings'].append(
                        f"Recovery used data quality fixes: {', '.join(quality_report['actions'])}"
                    )
                    result['processing_time'] = (datetime.now() - start_time).total_seconds()
                    return result
            
            # Strategy 2: Resample to lower frequency
            try:
                logger.info("Recovery strategy: Lower resampling frequency")
                
                # Copy with lower frequency
                lower_freq = min(self.resample_freq / 2, 100)  # Try half rate or max 100Hz
                
                processor = SequentialProcessor(
                    memory_manager=self.memory_manager,
                    montage=self.montage,
                    resample_freq=lower_freq,
                    lambda2=self.lambda2,
                    chunk_seconds=self.chunk_seconds
                )
                
                # Try processing
                modified_result = processor.process_file(input_file, output_dir)
                
                if modified_result['status'] == 'success':
                    self.recovery_stats['successful'] += 1
                    result.update(modified_result)
                    result['recovery_successful'] = True
                    result['warnings'].append(
                        f"Recovery used lower sampling frequency: {lower_freq}Hz"
                    )
                    result['processing_time'] = (datetime.now() - start_time).total_seconds()
                    return result
            except Exception as e:
                logger.warning(f"Lower frequency recovery failed: {e}")
            
            # Strategy failed
            raise ProcessingError(f"Data quality recovery strategies exhausted: {str(exception)}")
            
        except Exception as e:
            # Recovery failed
            self.recovery_stats['failed'] += 1
            
            result['status'] = 'failed'
            result['error'] = f"Data quality recovery failed: {str(e)}"
            result['original_error'] = str(exception)
            result['recovery_successful'] = False
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            # Save error report
            self._save_error_report(input_file, e, result, original_error=exception)
            
            return result
    
    def _recover_memory_issues(self, input_file: str, output_dir: str, 
                              exception: Exception, result: Dict[str, Any],
                              start_time: datetime) -> Dict[str, Any]:
        """Recovery strategy for memory issues."""
        strategy = "memory_recovery"
        self._count_strategy(strategy)
        result['recovery_strategy'] = strategy
        
        logger.info(f"Attempting memory issue recovery for {os.path.basename(input_file)}")
        
        try:
            # Strategy 1: Try with lower resampling frequency
            try:
                logger.info("Recovery strategy: Lower resampling frequency")
                
                # Copy with lower frequency
                lower_freq = min(self.resample_freq / 2, 100)  # Try half rate or max 100Hz
                
                processor = SequentialProcessor(
                    memory_manager=self.memory_manager,
                    montage=self.montage,
                    resample_freq=lower_freq,
                    lambda2=self.lambda2,
                    chunk_seconds=self.chunk_seconds
                )
                
                # Try processing
                modified_result = processor.process_file(input_file, output_dir)
                
                if modified_result['status'] == 'success':
                    self.recovery_stats['successful'] += 1
                    result.update(modified_result)
                    result['recovery_successful'] = True
                    result['warnings'].append(
                        f"Recovery used lower sampling frequency: {lower_freq}Hz"
                    )
                    result['processing_time'] = (datetime.now() - start_time).total_seconds()
                    return result
            except Exception as e:
                logger.warning(f"Lower frequency recovery failed: {e}")
            
            # Strategy 2: Try with epochs subsampling
            try:
                logger.info("Recovery strategy: Epoch subsampling")
                
                # Read epochs but only keep a subset
                import mne
                epochs = mne.io.read_epochs_eeglab(input_file, verbose=False)
                
                # Take at most 20 epochs
                max_epochs = min(len(epochs), 20)
                epochs = epochs[:max_epochs]
                
                # Process with subset of epochs
                modified_result = self._process_with_epochs(
                    epochs, output_dir, 
                    subject_id=os.path.splitext(os.path.basename(input_file))[0] + "_subset"
                )
                
                if modified_result['status'] == 'success':
                    self.recovery_stats['successful'] += 1
                    result.update(modified_result)
                    result['recovery_successful'] = True
                    result['warnings'].append(
                        f"Recovery used epoch subsampling: {max_epochs}/{len(epochs)} epochs"
                    )
                    result['processing_time'] = (datetime.now() - start_time).total_seconds()
                    return result
            except Exception as e:
                logger.warning(f"Epoch subsampling failed: {e}")
            
            # Strategy failed
            raise ProcessingError(f"Memory recovery strategies exhausted: {str(exception)}")
            
        except Exception as e:
            # Recovery failed
            self.recovery_stats['failed'] += 1
            
            result['status'] = 'failed'
            result['error'] = f"Memory recovery failed: {str(e)}"
            result['original_error'] = str(exception)
            result['recovery_successful'] = False
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            # Save error report
            self._save_error_report(input_file, e, result, original_error=exception)
            
            return result
    
    def _recover_generic(self, input_file: str, output_dir: str, 
                        exception: Exception, result: Dict[str, Any],
                        start_time: datetime) -> Dict[str, Any]:
        """Generic fallback recovery strategy."""
        strategy = "generic_recovery"
        self._count_strategy(strategy)
        result['recovery_strategy'] = strategy
        
        logger.info(f"Attempting generic recovery for {os.path.basename(input_file)}")
        
        try:
            # Try with very conservative settings
            try:
                logger.info("Recovery strategy: Conservative settings")
                
                # Copy with conservative settings
                processor = SequentialProcessor(
                    memory_manager=MemoryManager(max_memory_gb=2),  # Limit memory
                    montage=self.montage,
                    resample_freq=100,  # Lower frequency
                    lambda2=1.0/4.0,  # More regularization
                    chunk_seconds=self.chunk_seconds
                )
                
                # Try processing
                modified_result = processor.process_file(input_file, output_dir)
                
                if modified_result['status'] == 'success':
                    self.recovery_stats['successful'] += 1
                    result.update(modified_result)
                    result['recovery_successful'] = True
                    result['warnings'].append("Recovery used conservative settings")
                    result['processing_time'] = (datetime.now() - start_time).total_seconds()
                    return result
            except Exception as e:
                logger.warning(f"Conservative settings recovery failed: {e}")
            
            # Strategy failed
            raise ProcessingError(f"Generic recovery strategies exhausted: {str(exception)}")
            
        except Exception as e:
            # Recovery failed
            self.recovery_stats['failed'] += 1
            
            result['status'] = 'failed'
            result['error'] = f"Generic recovery failed: {str(e)}"
            result['original_error'] = str(exception)
            result['recovery_successful'] = False
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            # Save error report
            self._save_error_report(input_file, e, result, original_error=exception)
            
            return result
    
    def _process_with_epochs(self, epochs: mne.Epochs, output_dir: str, 
                            subject_id: str) -> Dict[str, Any]:
        """Process with pre-loaded epochs."""
        result = {
            'status': 'failed',
            'output_file': None,
            'error': None
        }
        
        try:
            # Pick EEG channels first to remove EOG, ECG, etc.
            logger.info("Selecting EEG channels only")
            
            # First set channel types for known EOG channels
            eog_channels = [ch for ch in epochs.ch_names if any(eog in ch.upper() for eog in ['EOG', 'HEOG', 'VEOG'])]
            if eog_channels:
                logger.info(f"Setting {len(eog_channels)} EOG channels: {eog_channels}")
                epochs.set_channel_types({ch: 'eog' for ch in eog_channels})
            
            # Now pick only EEG channels
            epochs.pick("eeg")
            
            # Set montage if needed
            if not epochs.get_montage():
                logger.info(f"Setting montage: {self.montage}")
                try:
                    epochs.set_montage(
                        mne.channels.make_standard_montage(self.montage), 
                        match_case=False
                    )
                except Exception as e:
                    logger.warning(f"Could not set montage: {e} - continuing without")
            
            # Resample if needed
            if epochs.info['sfreq'] != self.resample_freq:
                logger.info(f"Resampling to {self.resample_freq} Hz")
                epochs.resample(self.resample_freq)
            
            # Set EEG reference
            epochs.set_eeg_reference(projection=True)
            
            # Continue with source localization
            self._setup_fsaverage()
            
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
            logger.info("Applying inverse solution...")
            stcs = mne.minimum_norm.apply_inverse_epochs(
                epochs, inv, lambda2=self.lambda2, method="MNE", 
                pick_ori='normal', verbose=False
            )
            
            # Convert to EEG format
            _, output_file = self._convert_stc_to_eeg(
                stcs, output_dir, subject_id, original_epochs=epochs
            )
            
            # Success
            result['status'] = 'success'
            result['output_file'] = output_file
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result
    
    def _save_error_report(self, input_file: str, exception: Exception, 
                           result: Dict[str, Any], original_error: Exception = None) -> None:
        """Save detailed error report to file."""
        if not self.error_dir:
            return
        
        try:
            # Ensure directory exists
            os.makedirs(self.error_dir, exist_ok=True)
            
            # Create report
            basename = os.path.splitext(os.path.basename(input_file))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.error_dir, f"{basename}_error_{timestamp}.json")
            
            # Prepare error report
            error_report = {
                'file': input_file,
                'timestamp': datetime.now().isoformat(),
                'error': str(exception),
                'error_type': type(exception).__name__,
                'traceback': traceback.format_exc(),
                'result': result
            }
            
            if original_error:
                error_report['original_error'] = str(original_error)
                error_report['original_error_type'] = type(original_error).__name__
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(error_report, f, indent=2)
                
            logger.info(f"Error report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")
    
    def _count_strategy(self, strategy: str) -> None:
        """Count usage of recovery strategy."""
        if strategy in self.recovery_stats['strategies_used']:
            self.recovery_stats['strategies_used'][strategy] += 1
        else:
            self.recovery_stats['strategies_used'][strategy] = 1
    
    def process_batch(self, file_list: List[str], output_dir: str) -> List[Dict[str, Any]]:
        """
        Process a batch of files with error recovery.
        
        Parameters
        ----------
        file_list : list
            List of input .set files
        output_dir : str
            Output directory
            
        Returns
        -------
        results : list
            List of processing results
        """
        results = []
        
        # Process files one by one (sequential for robustness)
        for file_path in file_list:
            try:
                result = self.process_with_recovery(file_path, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Unhandled error in process_batch: {e}")
                results.append({
                    'input_file': file_path,
                    'status': 'failed',
                    'error': str(e),
                    'recovery_attempted': False
                })
        
        return results
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        stats = self.recovery_stats.copy()
        
        # Add success rate
        if stats['attempted'] > 0:
            stats['success_rate'] = (stats['successful'] / stats['attempted']) * 100
        else:
            stats['success_rate'] = 0
            
        return stats

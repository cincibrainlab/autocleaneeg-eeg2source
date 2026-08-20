"""Parallel processing pipeline for EEG to source conversion."""

import os
import gc
import time
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np
import mne
from mne.datasets import fetch_fsaverage
import pandas as pd
from functools import partial

from .converter import SequentialProcessor
from .memory_manager import MemoryManager
from ..io.exceptions import ProcessingError

logger = logging.getLogger(__name__)


# Helper function at module level for multiprocessing
def _process_batch_helper(file_path, output_dir, montage, resample_freq, lambda2, chunk_seconds):
    """Helper function for batch processing to avoid pickling issues.

    Receives the batch owner's actual montage/resample_freq/lambda2/
    chunk_seconds so per-file worker processes honor the caller's
    configuration instead of silently reverting to class defaults.
    ``n_jobs`` stays pinned to 1 here intentionally: file-level parallelism
    is already provided by process_batch's outer ProcessPoolExecutor, so
    each per-file worker must not itself spawn a nested worker pool.
    """
    try:
        # Create a new processor for this process to avoid shared state issues
        from .memory_manager import MemoryManager
        from .parallel_processor import ParallelProcessor

        processor = ParallelProcessor(
            memory_manager=MemoryManager(),  # Create a new memory manager
            montage=montage,
            resample_freq=resample_freq,
            lambda2=lambda2,
            n_jobs=1,                        # Use single thread within each file process
            batch_size=4,                    # Use default batch size
            parallel_method='processes',      # Use default parallel method
            chunk_seconds=chunk_seconds,
        )
        return processor.process_file(file_path, output_dir)
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(file_path)}: {e}")
        return {
            'input_file': file_path,
            'status': 'failed',
            'error': str(e)
        }


class ParallelProcessor(SequentialProcessor):
    """Parallel processor for EEG to source localization conversion."""
    
    def __init__(self, 
                 memory_manager: Optional[MemoryManager] = None,
                 montage: str = "GSN-HydroCel-129",
                 resample_freq: float = 250,
                 lambda2: float = 1.0 / 9.0,
                 n_jobs: int = 1,
                 batch_size: int = 4,
                 parallel_method: str = 'processes',
                 chunk_seconds: float = 30.0):
        """
        Initialize parallel processor.
        
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
        n_jobs : int
            Outer/label-processing workers. ``-1`` preserves the legacy
            all-available-cores behavior; the default is 1.
        batch_size : int
            Number of epochs to process in parallel
        parallel_method : str
            Method for parallelization ('processes' or 'threads')
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
        
        # Set number of parallel jobs
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
            raise ValueError("n_jobs must be -1 or a positive integer")
        if n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        elif n_jobs > 0:
            self.n_jobs = n_jobs
        else:
            raise ValueError("n_jobs must be -1 or a positive integer")
        self.batch_size = batch_size
        self.parallel_method = parallel_method
        
        # Performance metrics
        self.metrics = {
            'setup_time': 0,
            'read_time': 0,
            'forward_time': 0,
            'inverse_time': 0,
            'extract_time': 0,
            'total_time': 0
        }
        
        logger.info(f"Initialized parallel processor with {self.n_jobs} workers "
                   f"using {parallel_method} method")
    
    def _get_forward_solution(self, info: mne.Info) -> mne.Forward:
        """Get cached or compute forward solution with parallel processing."""
        if self.forward_solution is not None:
            # Check if montage matches
            if len(self.forward_solution['info']['ch_names']) == len(info['ch_names']):
                logger.debug("Using cached forward solution")
                return self.forward_solution
        
        logger.info("Computing forward solution...")
        self.memory_manager.check_available()
        
        # Compute forward solution with parallel processing
        self.forward_solution = mne.make_forward_solution(
            info, 
            trans="fsaverage", 
            src=self.fsaverage_src, 
            bem=self.fsaverage_bem,
            eeg=True, 
            mindist=5.0, 
            n_jobs=1  # Keep source-space work single-threaded per file
        )
        
        self.memory_manager.log_memory_status("After forward solution")
        return self.forward_solution
    
    def process_file(self, input_file: str, output_dir: str) -> Dict[str, Any]:
        """
        Process single EEG file with parallel processing.
        
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
        start_time = time.time()
        
        result = {
            'input_file': input_file,
            'status': 'failed',
            'output_file': None,
            'error': None,
            'metrics': None
        }
        
        try:
            # Reset metrics
            self.metrics = {key: 0 for key in self.metrics}
            
            # Validate input file
            logger.info(f"Processing: {os.path.basename(input_file)}")
            report = self.validator.validate_file_pair(input_file)
            if report['file_type'] == 'raw':
                return SequentialProcessor.process_file(self, input_file, output_dir)
            
            # Check memory before starting
            self.memory_manager.check_available()
            
            # Setup fsaverage if needed
            setup_start = time.time()
            self._setup_fsaverage()
            self.metrics['setup_time'] = time.time() - setup_start
            
            # Load epochs
            read_start = time.time()
            if report['file_type'] == 'epochs':
                epochs = self.reader.read_epochs(input_file)
            else:
                epochs = self.reader.read_raw(input_file)
            self.metrics['read_time'] = time.time() - read_start
            
            # Pick EEG channels first to remove EOG, ECG, etc.
            logger.info("Selecting EEG channels only")
            
            # First set channel types for known EOG channels
            eog_channels = [ch for ch in epochs.ch_names if any(eog in ch.upper() for eog in ['EOG', 'HEOG', 'VEOG'])]
            if eog_channels:
                logger.info(f"Setting {len(eog_channels)} EOG channels: {eog_channels}")
                epochs.set_channel_types({ch: 'eog' for ch in eog_channels})
            
            # Now pick only EEG channels
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
            forward_start = time.time()
            fwd = self._get_forward_solution(epochs.info)
            self.metrics['forward_time'] = time.time() - forward_start
            
            # Compute noise covariance
            logger.info("Computing noise covariance...")
            noise_cov = mne.make_ad_hoc_cov(epochs.info)
            
            # Create inverse operator
            logger.info("Creating inverse operator...")
            inv = mne.minimum_norm.make_inverse_operator(
                epochs.info, fwd, noise_cov, verbose=False
            )
            
            # Apply inverse solution to epochs in parallel
            inverse_start = time.time()
            stcs = self._apply_inverse_parallel(epochs, inv)
            self.metrics['inverse_time'] = time.time() - inverse_start
            
            # Convert to EEG format with DK regions using parallel processing
            extract_start = time.time()
            output_epochs, output_file = self._convert_stc_to_eeg_parallel(
                stcs, output_dir, 
                subject_id=os.path.splitext(os.path.basename(input_file))[0],
                original_epochs=epochs
            )
            self.metrics['extract_time'] = time.time() - extract_start
            
            # Update result
            result['status'] = 'success'
            result['output_file'] = output_file
            
            # Cleanup
            del epochs, inv, stcs, output_epochs
            gc.collect()
            self.memory_manager.cleanup()
            
            # Set total processing time
            self.metrics['total_time'] = time.time() - start_time
            result['metrics'] = self.metrics.copy()
            
            logger.info(f"✓ Successfully processed: {output_file} in {self.metrics['total_time']:.2f}s")
            
        except MemoryError as e:
            logger.error(f"Memory exhausted: {e}")
            result['error'] = str(e)
            # Try to cleanup
            gc.collect()
            self.memory_manager.cleanup()
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            result['error'] = str(e)
            
        return result
    
    def _apply_inverse_parallel(self, epochs: mne.Epochs, inv: mne.minimum_norm.InverseOperator) -> List:
        """Apply inverse solution to epochs using parallel processing."""
        n_epochs = len(epochs)
        logger.info(f"Applying inverse solution to {n_epochs} epochs in parallel...")
        
        # Split epochs into batches to optimize memory usage
        batch_size = min(self.batch_size, n_epochs)
        n_batches = int(np.ceil(n_epochs / batch_size))
        
        all_stcs = []
        
        # Process each batch
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_epochs)
            logger.debug(f"Processing batch {batch_idx+1}/{n_batches} (epochs {start_idx}-{end_idx})")
            
            # Get batch of epochs
            batch_epochs = epochs[start_idx:end_idx]
            
            # Use apply_inverse_epochs (note: this function doesn't accept n_jobs parameter)
            batch_stcs = mne.minimum_norm.apply_inverse_epochs(
                batch_epochs, inv, lambda2=self.lambda2, method="MNE", 
                pick_ori='normal', verbose=False
            )
            
            all_stcs.extend(batch_stcs)
            
            # Check memory
            self.memory_manager.check_available()
        
        return all_stcs
    
    def _process_stc(self, stc):
        """Process a single source time course (helper method)."""
        # Extract label time courses
        return mne.extract_label_time_course(
            stc, self.labels, src=self.fsaverage_src, 
            mode='mean', verbose=False
        )
        
    def _convert_stc_to_eeg_parallel(self, stc_list: list, output_dir: str, subject_id: str, original_epochs: mne.Epochs = None) -> tuple:
        """Convert source estimates to EEG format with DK atlas regions using parallel processing."""
        logger.info(f"Converting {len(stc_list)} source estimates to EEG format in parallel...")
        
        # Always use ThreadPoolExecutor for this operation to avoid pickling issues
        # with class methods that access instance variables (self.labels, self.fsaverage_src)
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            all_label_ts = list(executor.map(self._process_stc, stc_list))
        
        # Stack to get 3D array (n_epochs, n_regions, n_times)
        label_data = np.array(all_label_ts)
        
        # Get properties
        n_epochs = len(stc_list)
        n_regions = len(self.labels)
        n_times = stc_list[0].data.shape[1]
        sfreq = 1.0 / stc_list[0].tstep
        ch_names = [label.name for label in self.labels]
        
        # Create channel positions
        ch_pos = {}
        for i, label in enumerate(self.labels):
            # Create positions on a sphere (simplified)
            theta = 2 * np.pi * i / n_regions
            phi = np.pi * (i % 4) / 4
            ch_pos[label.name] = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ]) * 0.1
        
        # Create MNE Info
        info = mne.create_info(
            ch_names=ch_names, 
            sfreq=sfreq, 
            ch_types=['eeg'] * n_regions
        )
        
        # Update channel positions
        for idx, ch_name in enumerate(ch_names):
            info['chs'][idx]['loc'][:3] = ch_pos[ch_name]
        
        # Create epochs with proper event handling
        if original_epochs is not None and hasattr(original_epochs, 'event_id') and hasattr(original_epochs, 'events'):
            # Preserve original event structure
            event_id = original_epochs.event_id.copy()
            # Use proper sample indices based on epoch timing to avoid EEGLABIO issues
            if len(original_epochs.events) == n_epochs:
                # Create realistic sample indices with padding to avoid EEGLABIO issues
                events = []
                epoch_duration = stc_list[0].times[-1] - stc_list[0].times[0]
                epoch_length_samples = int(sfreq * epoch_duration)
                # Add padding between epochs to prevent EEGLABIO from adding dummy events
                padding_samples = int(sfreq * 0.1)  # 100ms padding
                epoch_spacing = epoch_length_samples + padding_samples
                
                for i in range(n_epochs):
                    sample_idx = i * epoch_spacing
                    events.append([sample_idx, 0, original_epochs.events[i, 2]])
                events = np.array(events)
            else:
                # Fallback: use the first event code from original data
                first_event_code = list(event_id.values())[0]
                events = []
                epoch_duration = stc_list[0].times[-1] - stc_list[0].times[0]
                epoch_length_samples = int(sfreq * epoch_duration)
                padding_samples = int(sfreq * 0.1)  # 100ms padding
                epoch_spacing = epoch_length_samples + padding_samples
                
                for i in range(n_epochs):
                    sample_idx = i * epoch_spacing
                    events.append([sample_idx, 0, first_event_code])
                events = np.array(events)
        else:
            # Default fallback
            events = []
            epoch_duration = stc_list[0].times[-1] - stc_list[0].times[0]
            epoch_length_samples = int(sfreq * epoch_duration)
            padding_samples = int(sfreq * 0.1)  # 100ms padding
            epoch_spacing = epoch_length_samples + padding_samples
            
            for i in range(n_epochs):
                sample_idx = i * epoch_spacing
                events.append([sample_idx, 0, 1])
            events = np.array(events)
            event_id = {'event': 1}
        tmin = stc_list[0].tmin
        
        epochs = mne.EpochsArray(
            label_data, info, events=events, 
            event_id=event_id, tmin=tmin
        )
        
        # Save to EEGLAB format
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{subject_id}_dk_regions.set")
        epochs.export(output_file, fmt='eeglab', overwrite=True)
        
        logger.info(f"Saved {n_regions} regions to {output_file}")
        
        # Save metadata
        self._save_metadata(output_dir, subject_id, ch_names, ch_pos)
        
        return epochs, output_file
    
    def process_batch(self, file_list: List[str], output_dir: str, 
                     max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process multiple files in parallel.
        
        Parameters
        ----------
        file_list : List[str]
            List of input .set files
        output_dir : str
            Output directory
        max_workers : int, optional
            Maximum number of parallel workers
            
        Returns
        -------
        results : List[Dict[str, Any]]
            Processing results for each file
        """
        logger.info(f"Processing {len(file_list)} files in batch mode")
        
        # Resolve the public all-core sentinel before the executor boundary.
        if max_workers is None or max_workers == -1:
            max_workers = self.n_jobs
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers <= 0:
            raise ValueError("max_workers must be -1 or a positive integer")
        
        # For process-based parallelism, we need to pass both file_path and output_dir
        # Use module-level helper function with output_dir, plus this processor's
        # actual montage/resample_freq/lambda2/chunk_seconds so batch workers do
        # not silently fall back to class defaults.
        process_func = partial(
            _process_batch_helper,
            output_dir=output_dir,
            montage=self.montage,
            resample_freq=self.resample_freq,
            lambda2=self.lambda2,
            chunk_seconds=self.chunk_seconds,
        )
        
        # Process files in parallel using module-level function
        results = []
        
        # Use process-based parallelism for file-level parallelism
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_func, file_path) for file_path in file_list]
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel processing: {str(e)}")
                    results.append({
                        'status': 'failed',
                        'error': str(e)
                    })
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance metrics report."""
        return {
            'metrics': self.metrics,
            'n_jobs': self.n_jobs,
            'parallel_method': self.parallel_method,
            'batch_size': self.batch_size
        }


class CachedProcessor(ParallelProcessor):
    """Processor with caching for intermediate results."""
    
    def __init__(self, 
                 memory_manager: Optional[MemoryManager] = None,
                 montage: str = "GSN-HydroCel-129",
                 resample_freq: float = 250,
                 lambda2: float = 1.0 / 9.0,
                 n_jobs: int = 1,
                 batch_size: int = 4,
                 parallel_method: str = 'processes',
                 cache_dir: Optional[str] = None,
                 chunk_seconds: float = 30.0):
        """
        Initialize cached processor.
        
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
        n_jobs : int
            Positive number of parallel jobs; defaults conservatively to 1
        batch_size : int
            Number of epochs to process in parallel
        parallel_method : str
            Method for parallelization ('processes' or 'threads')
        cache_dir : str, optional
            Directory for caching intermediate results
        chunk_seconds : float, optional
            Continuous/raw inverse chunk length in seconds
        """
        super().__init__(
            memory_manager=memory_manager,
            montage=montage,
            resample_freq=resample_freq,
            lambda2=lambda2,
            chunk_seconds=chunk_seconds,
            n_jobs=n_jobs,
            batch_size=batch_size,
            parallel_method=parallel_method
        )
        
        # Setup caching
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Caching enabled with cache directory: {cache_dir}")
        else:
            logger.info("Caching disabled")
        
        # Cache hit/miss metrics
        self.cache_metrics = {
            'forward_hits': 0,
            'forward_misses': 0,
            'inverse_hits': 0,
            'inverse_misses': 0
        }
    
    def _get_cache_path(self, prefix: str, identifier: str, suffix: str = 'fif') -> str:
        """Get path for cached file."""
        if not self.cache_dir:
            return None
            
        return os.path.join(self.cache_dir, f"{prefix}_{identifier}.{suffix}")
    
    def _get_forward_solution(self, info: mne.Info) -> mne.Forward:
        """Get cached or compute forward solution with caching."""
        if not self.cache_dir:
            return super()._get_forward_solution(info)
        
        # Create cache identifier based on channel names and positions
        ch_names = info['ch_names']
        ch_names_str = "_".join(ch_names)
        import hashlib
        identifier = hashlib.md5(ch_names_str.encode()).hexdigest()
        
        # Check if cached forward solution exists
        cache_path = self._get_cache_path('fwd', identifier)
        
        if os.path.exists(cache_path):
            logger.info(f"Loading forward solution from cache: {cache_path}")
            self.forward_solution = mne.read_forward_solution(cache_path)
            self.cache_metrics['forward_hits'] += 1
            return self.forward_solution
        
        # Compute new forward solution
        self.cache_metrics['forward_misses'] += 1
        fwd = super()._get_forward_solution(info)
        
        # Save to cache
        mne.write_forward_solution(cache_path, fwd, overwrite=True)
        logger.info(f"Saved forward solution to cache: {cache_path}")
        
        return fwd
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache hit/miss metrics."""
        metrics = self.cache_metrics.copy()
        
        # Calculate hit rates
        if metrics['forward_hits'] + metrics['forward_misses'] > 0:
            metrics['forward_hit_rate'] = (metrics['forward_hits'] / 
                                         (metrics['forward_hits'] + metrics['forward_misses']))
        else:
            metrics['forward_hit_rate'] = 0
            
        if metrics['inverse_hits'] + metrics['inverse_misses'] > 0:
            metrics['inverse_hit_rate'] = (metrics['inverse_hits'] / 
                                         (metrics['inverse_hits'] + metrics['inverse_misses']))
        else:
            metrics['inverse_hit_rate'] = 0
            
        return metrics

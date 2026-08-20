"""Benchmarking tools for performance measurement."""

import os
import time
import json
import logging
import platform
import datetime
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import numpy as np
import psutil

from ..core.converter import SequentialProcessor
from ..core.parallel_processor import ParallelProcessor, CachedProcessor
from ..core.gpu_processor import GPUProcessor
from ..core.memory_manager import MemoryManager
from ..core.optimized_memory import OptimizedMemoryManager

logger = logging.getLogger(__name__)


class BenchmarkTimer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "operation"):
        """
        Initialize timer.
        
        Parameters
        ----------
        name : str
            Name of the operation being timed
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing when entering context."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing when exiting context."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {duration:.3f} seconds")
    
    def duration(self) -> float:
        """Get the duration of the timed operation."""
        if self.start_time is None:
            return 0
        end = self.end_time or time.time()
        return end - self.start_time


class PerformanceBenchmark:
    """Performance benchmarking for EEG to source processing."""
    
    def __init__(self, 
                 output_dir: str = "./benchmark_results",
                 save_results: bool = True):
        """
        Initialize benchmark.
        
        Parameters
        ----------
        output_dir : str
            Directory to save benchmark results
        save_results : bool
            Whether to save results to file
        """
        self.output_dir = output_dir
        self.save_results = save_results
        
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        # Store benchmark results
        self.results = []
        self.current_run = {}
        
        logger.info(f"Initialized performance benchmark with output_dir={output_dir}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        cpu_info = {}
        
        # Try to get detailed CPU info on Linux
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
        except ImportError:
            # Fallback to basic info
            cpu_info = {
                'brand_raw': platform.processor(),
                'count': psutil.cpu_count(logical=True),
                'physical_count': psutil.cpu_count(logical=False)
            }
        
        # Memory info
        mem = psutil.virtual_memory()
        
        # GPU info
        gpu_info = {}
        try:
            from ..core.gpu_processor import check_gpu_availability
            gpu_info = check_gpu_availability()
        except Exception:
            gpu_info = {'gpu_count': 0}
        
        return {
            'os': platform.system(),
            'os_version': platform.version(),
            'python_version': platform.python_version(),
            'cpu': cpu_info,
            'memory_total_gb': mem.total / (1024**3),
            'memory_available_gb': mem.available / (1024**3),
            'gpu': gpu_info,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def setup_benchmark_run(self, 
                           test_name: str,
                           test_description: str = "",
                           processor_type: str = "sequential",
                           test_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Setup a new benchmark run.
        
        Parameters
        ----------
        test_name : str
            Name of the benchmark test
        test_description : str
            Description of what is being tested
        processor_type : str
            Type of processor being benchmarked
        test_params : dict, optional
            Additional test parameters
        """
        self.current_run = {
            'test_name': test_name,
            'test_description': test_description,
            'processor_type': processor_type,
            'system_info': self.get_system_info(),
            'test_params': test_params or {},
            'results': [],
            'start_time': time.time(),
            'end_time': None,
            'total_duration': None,
            'summary_stats': {}
        }
        
        logger.info(f"Starting benchmark: {test_name}")
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Add a single result to the current benchmark run.
        
        Parameters
        ----------
        result : dict
            Result data from a single test
        """
        if not self.current_run:
            logger.warning("No active benchmark run. Call setup_benchmark_run first.")
            return
        
        self.current_run['results'].append(result)
    
    def complete_benchmark_run(self) -> Dict[str, Any]:
        """
        Complete the current benchmark run, calculate statistics, and save results.
        
        Returns
        -------
        dict
            The completed benchmark run data
        """
        if not self.current_run:
            logger.warning("No active benchmark run to complete.")
            return {}
        
        # Set end time and calculate duration
        self.current_run['end_time'] = time.time()
        self.current_run['total_duration'] = (
            self.current_run['end_time'] - self.current_run['start_time']
        )
        
        # Calculate summary statistics
        self.current_run['summary_stats'] = self._calculate_summary_stats(
            self.current_run['results']
        )
        
        # Add to results list
        self.results.append(self.current_run)
        
        # Save results if enabled
        if self.save_results:
            self._save_benchmark_results(self.current_run)
        
        logger.info(
            f"Benchmark complete: {self.current_run['test_name']} in "
            f"{self.current_run['total_duration']:.2f}s"
        )
        
        # Return a copy to avoid modification
        result = self.current_run.copy()
        self.current_run = {}
        return result
    
    def _calculate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from a list of results."""
        if not results:
            return {}
        
        # Extract timing metrics
        metrics = {}
        
        # Collect metrics across all results
        for result in results:
            if 'metrics' in result:
                for key, value in result['metrics'].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            
            if 'gpu_metrics' in result:
                for key, value in result.get('gpu_metrics', {}).items():
                    gpu_key = f"gpu_{key}"
                    if gpu_key not in metrics:
                        metrics[gpu_key] = []
                    metrics[gpu_key].append(value)
        
        # Calculate statistics for each metric
        stats = {}
        for key, values in metrics.items():
            if not values:
                continue
                
            values = np.array(values)
            stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return stats
    
    def _save_benchmark_results(self, run: Dict[str, Any]) -> None:
        """Save benchmark results to a file."""
        if not self.output_dir:
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (f"{run['test_name'].replace(' ', '_').lower()}_"
                   f"{run['processor_type']}_{timestamp}.json")
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(run, f, indent=2)
            
            logger.info(f"Saved benchmark results to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
    
    def benchmark_processor(self, 
                           processor: Union[SequentialProcessor, ParallelProcessor, GPUProcessor],
                           input_files: List[str],
                           output_dir: str,
                           test_name: Optional[str] = None,
                           test_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Benchmark a processor's performance on a set of input files.
        
        Parameters
        ----------
        processor : SequentialProcessor or ParallelProcessor or GPUProcessor
            The processor to benchmark
        input_files : list of str
            List of input files to process
        output_dir : str
            Output directory for processed files
        test_name : str, optional
            Name for the benchmark test
        test_description : str, optional
            Description of the benchmark test
            
        Returns
        -------
        dict
            Benchmark results
        """
        # Determine processor type
        if isinstance(processor, GPUProcessor):
            processor_type = "gpu"
        elif isinstance(processor, CachedProcessor):
            processor_type = "cached"
        elif isinstance(processor, ParallelProcessor):
            processor_type = "parallel"
        else:
            processor_type = "sequential"
        
        # Setup test name if not provided
        if test_name is None:
            test_name = f"{processor_type}_processor_test"
        
        # Setup description if not provided
        if test_description is None:
            test_description = f"Benchmark for {processor_type} processor with {len(input_files)} files"
        
        # Get processor parameters
        processor_params = {
            'montage': processor.montage,
            'resample_freq': processor.resample_freq,
            'lambda2': processor.lambda2
        }
        
        # Add parallel-specific parameters
        if hasattr(processor, 'n_jobs'):
            processor_params['n_jobs'] = processor.n_jobs
            processor_params['batch_size'] = processor.batch_size
            processor_params['parallel_method'] = getattr(processor, 'parallel_method', None)
        
        # Add GPU-specific parameters
        if hasattr(processor, 'gpu_backend'):
            processor_params['gpu_backend'] = processor.gpu_backend
        
        # Setup benchmark run
        self.setup_benchmark_run(
            test_name=test_name,
            test_description=test_description,
            processor_type=processor_type,
            test_params=processor_params
        )
        
        # Process each file and record results
        for input_file in input_files:
            logger.info(f"Benchmarking {os.path.basename(input_file)}")
            
            try:
                with BenchmarkTimer(f"Processing {os.path.basename(input_file)}"):
                    result = processor.process_file(input_file, output_dir)
                
                # Add filename to result
                result['filename'] = os.path.basename(input_file)
                
                # Add to benchmark results
                self.add_result(result)
                
            except Exception as e:
                logger.error(f"Error during benchmark: {e}")
                self.add_result({
                    'filename': os.path.basename(input_file),
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Complete benchmark and return results
        return self.complete_benchmark_run()
    
    def compare_processors(self, 
                          input_files: List[str],
                          output_dir: str,
                          processors: Dict[str, Union[SequentialProcessor, 
                                                   ParallelProcessor, 
                                                   GPUProcessor]]) -> Dict[str, Any]:
        """
        Compare multiple processors on the same input files.
        
        Parameters
        ----------
        input_files : list of str
            List of input files to process
        output_dir : str
            Output directory for processed files
        processors : dict
            Dictionary mapping processor names to processor instances
            
        Returns
        -------
        dict
            Comparison results
        """
        comparison = {
            'test_name': 'processor_comparison',
            'description': f"Comparing {len(processors)} processors on {len(input_files)} files",
            'timestamp': datetime.datetime.now().isoformat(),
            'system_info': self.get_system_info(),
            'results': {}
        }
        
        # Create separate output dirs for each processor
        for processor_name in processors:
            processor_output_dir = os.path.join(output_dir, processor_name)
            os.makedirs(processor_output_dir, exist_ok=True)
        
        # Benchmark each processor
        for processor_name, processor in processors.items():
            logger.info(f"Benchmarking processor: {processor_name}")
            
            processor_output_dir = os.path.join(output_dir, processor_name)
            
            try:
                # Run benchmark
                result = self.benchmark_processor(
                    processor=processor,
                    input_files=input_files,
                    output_dir=processor_output_dir,
                    test_name=f"{processor_name}_test",
                    test_description=f"Benchmark for {processor_name}"
                )
                
                # Add to comparison results
                comparison['results'][processor_name] = result
                
            except Exception as e:
                logger.error(f"Error benchmarking {processor_name}: {e}")
                comparison['results'][processor_name] = {'error': str(e)}
        
        # Calculate speedups
        if len(processors) > 1 and 'sequential' in comparison['results']:
            # Use sequential as baseline
            baseline = comparison['results']['sequential']
            baseline_time = baseline.get('summary_stats', {}).get('total_time', {}).get('mean', 0)
            
            if baseline_time > 0:
                speedups = {}
                for name, result in comparison['results'].items():
                    if name == 'sequential':
                        speedups[name] = 1.0
                    else:
                        time = result.get('summary_stats', {}).get('total_time', {}).get('mean', 0)
                        if time > 0:
                            speedups[name] = baseline_time / time
                        else:
                            speedups[name] = 0
                
                comparison['speedups'] = speedups
        
        # Save comparison results
        if self.save_results:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"processor_comparison_{timestamp}.json")
            
            try:
                with open(filepath, 'w') as f:
                    json.dump(comparison, f, indent=2)
                
                logger.info(f"Saved comparison results to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save comparison results: {e}")
        
        return comparison


def run_standard_benchmark(input_files: List[str], 
                          output_dir: str,
                          n_jobs: int = 1,
                          enable_cache: bool = True,
                          enable_gpu: bool = True,
                          max_memory_gb: float = 4.0) -> Dict[str, Any]:
    """
    Run a standard benchmark comparing all processor types.
    
    Parameters
    ----------
    input_files : list of str
        List of input files to process
    output_dir : str
        Output directory for processed files
    n_jobs : int
        Positive number of parallel jobs; defaults conservatively to 1
    enable_cache : bool
        Whether to enable caching
    enable_gpu : bool
        Whether to enable GPU processing
    max_memory_gb : float
        Maximum memory usage in GB
        
    Returns
    -------
    dict
        Benchmark results
    """
    benchmark = PerformanceBenchmark(
        output_dir=os.path.join(output_dir, "benchmark_results")
    )
    
    # Create processors
    processors = {}
    
    # Sequential processor (baseline)
    memory_manager = MemoryManager(max_memory_gb=max_memory_gb)
    processors['sequential'] = SequentialProcessor(
        memory_manager=memory_manager
    )
    
    # Optimized memory processor
    opt_memory_manager = OptimizedMemoryManager(
        max_memory_gb=max_memory_gb,
        enable_disk_offload=True,
        enable_auto_cleanup=True
    )
    processors['optimized_memory'] = SequentialProcessor(
        memory_manager=opt_memory_manager
    )
    
    # Parallel processor
    processors['parallel'] = ParallelProcessor(
        memory_manager=memory_manager,
        n_jobs=n_jobs,
        batch_size=4,
        parallel_method='processes'
    )
    
    # Cached processor
    if enable_cache:
        cache_dir = os.path.join(output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        processors['cached'] = CachedProcessor(
            memory_manager=memory_manager,
            n_jobs=n_jobs,
            batch_size=4,
            parallel_method='processes',
            cache_dir=cache_dir
        )
    
    # GPU processor
    if enable_gpu:
        from ..core.gpu_processor import check_gpu_availability
        gpu_info = check_gpu_availability()
        
        if gpu_info['gpu_count'] > 0:
            processors['gpu'] = GPUProcessor(
                memory_manager=memory_manager,
                n_jobs=n_jobs,
                batch_size=4,
                parallel_method='processes',
                gpu_backend='auto'
            )
    
    # Run comparison
    comparison = benchmark.compare_processors(
        input_files=input_files,
        output_dir=output_dir,
        processors=processors
    )
    
    # Clean up
    if enable_cache:
        processors['cached'].memory_manager.cleanup()
    
    if 'optimized_memory' in processors:
        opt_memory_manager.stop()
    
    return comparison


if __name__ == "__main__":
    # Example usage (for documentation)
    from ..core.converter import SequentialProcessor
    from ..core.parallel_processor import ParallelProcessor
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Simple benchmark
    benchmark = PerformanceBenchmark()
    
    # Create processors
    sequential = SequentialProcessor()
    parallel = ParallelProcessor(n_jobs=4)
    
    # Input files
    input_files = ["file1.set", "file2.set"]
    
    # Compare processors
    comparison = benchmark.compare_processors(
        input_files=input_files,
        output_dir="./output",
        processors={
            'sequential': sequential,
            'parallel': parallel
        }
    )

"""AutoClean EEG2Source: EEG source localization with DK atlas regions."""

__version__ = "0.4.0"
__author__ = "AutoClean Team"
RAW_CHUNKING_CAPABILITY = "sequential-raw-chunking-v1"

from .core.converter import SequentialProcessor
from .core.memory_manager import MemoryManager
from .core.robust_processor import RobustProcessor
from .core.continuous_processor import ContinuousProcessor
from .io.eeglab_reader import EEGLABReader
from .io.validators import EEGLABValidator
from .io.data_quality import QualityAssessor
from .utils.error_reporter import ErrorReporter, ErrorHandler
from .utils.logging import setup_logger

__all__ = [
    # Core processing
    "SequentialProcessor",
    "RobustProcessor",
    "ContinuousProcessor",
    "MemoryManager",
    "RAW_CHUNKING_CAPABILITY",
    
    # IO and validation
    "EEGLABReader",
    "EEGLABValidator",
    "QualityAssessor",
    
    # Utilities
    "ErrorReporter",
    "ErrorHandler",
    "setup_logger",
]

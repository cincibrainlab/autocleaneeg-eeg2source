# AutoClean EEG2Source

EEG source localization with Desikan-Killiany (DK) atlas regions. This package converts EEG epochs to source-localized data using the DK brain atlas.

> **Built on MNE-Python**: AutoClean EEG2Source is powered by [MNE-Python](https://mne.tools/stable/index.html), the leading open-source Python package for analyzing EEG, MEG, and other neurophysiological data. We gratefully acknowledge the MNE development team for creating this exceptional foundation for neurophysiological data analysis.

## Overview

AutoClean EEG2Source is a specialized tool designed for processing high-density EEG data into source-localized representations using brain regions from the Desikan-Killiany atlas. The tool provides a robust, memory-efficient pipeline with parallel processing capabilities, GPU acceleration, and automatic error recovery.

## Features

- Convert EEG epochs to source-localized data with DK atlas regions
- Memory-efficient processing with monitoring
- Support for EEGLAB .set file format
- Batch processing capabilities
- Robust error recovery mechanisms
- Parallel processing for improved performance
- GPU acceleration for supported operations
- Caching of intermediate results
- Comprehensive benchmarking tools
- Command-line interface

## Technical Requirements and Assumptions

### Hardware Requirements

- **Minimum**: 
  - 8GB RAM
  - 4 CPU cores
  - 10GB free disk space
- **Recommended**:
  - 16GB+ RAM
  - 8+ CPU cores
  - 50GB+ free disk space
  - NVIDIA GPU with CUDA support (for GPU acceleration)

### Software Requirements

- **Operating System**: 
  - Linux (tested on Ubuntu 20.04+)
  - macOS (10.15+)
  - Windows 10+ (with limitations)
- **Python**: 3.8 or higher
- **Dependencies**:
  - MNE-Python 1.6.0 (strict dependency)
  - nibabel
  - numpy
  - pandas
  - loguru
  - psutil
  - eeglabio
  - joblib (for parallel processing)
  - Optional: cupy, pytorch, or tensorflow (for GPU acceleration)

### EEG Data Assumptions

#### Supported EEG Systems and Formats

This tool was **primarily designed for high-density EEG systems**, particularly:
- **EGI/Philips** high-density nets (64, 128, 256 channel systems)
- **BioSemi** ActiveTwo systems
- Other systems with standardized 10-20, 10-10, or 10-5 electrode placement

The EEGLAB `.set/.fdt` format is the only currently supported file format. The tool assumes data has been preprocessed and epoched using EEGLAB or a compatible tool.

#### Montage Compatibility

The default montage is **"GSN-HydroCel-129"** (EGI 128-channel net). Other supported montages include:
- Standard montages: "standard_1005", "standard_1020"
- EGI/Philips: "GSN-HydroCel-32", "GSN-HydroCel-64", "GSN-HydroCel-128", "GSN-HydroCel-256"
- BioSemi: "biosemi16", "biosemi32", "biosemi64"
- EasyCap: "easycap-M1"

Montage mismatch will be automatically detected during validation, and the robust processor will attempt montage recovery.

#### Data Quality and Preprocessing

The tool assumes data that has undergone standard preprocessing steps:
- Filtering (typically 1-40Hz bandpass)
- Artifact rejection (eye, muscle, and other artifacts removed)
- ICA for artifact removal (recommended)
- Segmentation into epochs
- Bad channel interpolation

Poor quality data may yield unreliable source estimates. The tool includes quality assessment features but cannot substitute for proper preprocessing.

### Source Localization Methodology

#### Forward Model Assumptions

- **Head model**: FSAverage template (MNI space)
- **Source space**: Surface-based, downsampled to the Desikan-Killiany atlas regions
- **Forward solution**: BEM (Boundary Element Method)
- **Conductivity model**: 3-shell (scalp, skull, brain)

The tool assumes the user's EEG data can be reasonably mapped to the FSAverage template. Individual head models are not supported.

#### Inverse Solution Parameters

- **Method**: Minimum Norm Estimate (MNE)
- **Regularization parameter**: λ² = 1/9 (default, can be customized)
- **Source orientation**: Normal to cortical surface

These defaults are based on common practices in the literature but may not be optimal for all research questions.

#### Output Representation

- 68 ROIs from the Desikan-Killiany atlas (34 per hemisphere)
- Time courses represent average activity within each region
- "_region_info.csv" file provides metadata on each region

## Installation

### Install with UV (Recommended)

UV is a blazing-fast Python package installer and resolver that significantly improves installation speed and reliability. It's the recommended way to install and run AutoClean EEG2Source:

```bash
# Install UV if you don't have it
pip install uv

# Install the package with UV
uv pip install autocleaneeg-eeg2source

# Or install from source with UV
uv pip install .

# Development installation
uv pip install -e .

# With development dependencies
uv pip install -e ".[dev]"
```

### Install with traditional pip

```bash
pip install autocleaneeg-eeg2source
```

### Install from source

```bash
pip install .
```

### Install in development mode

```bash
pip install -e .
```

### Install with development dependencies

```bash
pip install -e ".[dev]"
```

## Command-Line Usage

The package provides a command-line interface for processing EEG files.

### Basic Process Command

Convert EEG epochs to source-localized data:

```bash
autocleaneeg-eeg2source process input.set --output-dir ./results
```

Process multiple files in a directory:

```bash
autocleaneeg-eeg2source process ./data --output-dir ./results --recursive
```

### Validation Command

Check if EEG files are valid:

```bash
autocleaneeg-eeg2source validate ./data
```

Check montage compatibility:

```bash
autocleaneeg-eeg2source validate ./data --check-montage --montage "GSN-HydroCel-129"
```

### Information Command

Display information about an EEG file:

```bash
autocleaneeg-eeg2source info input.set
```

### Performance Options

#### Robust Processing (Error Recovery)

```bash
autocleaneeg-eeg2source process input.set --robust --error-dir ./errors
```

#### Parallel Processing

```bash
autocleaneeg-eeg2source process ./data --parallel --n-jobs 4 --batch-processing
```

#### Memory Optimization

```bash
autocleaneeg-eeg2source process input.set --optimized-memory --disk-offload --max-memory 8.0
```

#### Caching

```bash
autocleaneeg-eeg2source process ./data --enable-cache
```

#### GPU Acceleration

```bash
autocleaneeg-eeg2source process input.set --gpu --gpu-backend auto
```

### Benchmarking

```bash
autocleaneeg-eeg2source benchmark ./data --test-all --max-files 3
```

## Python API Usage

```python
from autoclean_eeg2source import SequentialProcessor, MemoryManager

# Initialize components
memory_manager = MemoryManager(max_memory_gb=4)
processor = SequentialProcessor(
    memory_manager=memory_manager,
    montage="GSN-HydroCel-129",
    resample_freq=250
)

# Process a file
result = processor.process_file("input.set", "./output")

if result['status'] == 'success':
    print(f"Output saved to: {result['output_file']}")
else:
    print(f"Processing failed: {result['error']}")
```

## Output Format

The package outputs:
- `.set` files with DK atlas regions as channels (68 regions)
- `_region_info.csv` with region metadata (names, hemispheres, positions)

## Limitations and Known Issues

1. **Template-based approach**: The tool uses a template head model (FSAverage) rather than individual head models, which may reduce accuracy for participants with atypical head shapes.

2. **Channel locations**: Accurate electrode positions are essential. If your actual electrode positions differ significantly from the standard montage, source estimates may be inaccurate.

3. **Memory constraints**: High-density EEG with many epochs requires substantial RAM. Use the memory optimization features for large datasets.

4. **GPU acceleration**: Current implementation offers limited GPU acceleration for specific operations and requires compatible hardware and software.

5. **Sampling rate**: Very high sampling rates (>1000Hz) may lead to excessive memory usage. Consider downsampling during preprocessing or using the `--resample-freq` option.

6. **Montage inference**: Automatic montage detection is best-effort and may not always succeed. Explicitly specify your montage when possible.

## Frequently Asked Questions

### Can I use this with my custom electrode layout?
Yes, if your channel names match one of the supported standard montages. For completely custom layouts, results may be less reliable.

### How do I interpret the source-localized data?
The output contains time courses for 68 regions of the Desikan-Killiany atlas. Each channel represents the average activity in that brain region.

### Can I use this for clinical diagnosis?
No. This tool is designed for research purposes only and has not been validated for clinical applications.

### How do I cite this tool?
Please cite the GitHub repository and the relevant MNE-Python publications as this tool builds upon the MNE framework.

### Is individual MRI required?
No. The tool uses the FSAverage template. While individual MRIs would improve accuracy, they are not supported in this implementation.

### How does the tool handle reference electrodes?
EEG data is re-referenced to average reference during processing.

## References and Citations

When using this tool, please cite:

1. The AutoClean EEG2Source repository
2. **MNE-Python**: Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7, 267. https://doi.org/10.3389/fnins.2013.00267
3. **MNE-Python Inverse Imaging**: Gramfort, A., et al. (2014). MNE software for processing MEG and EEG data. NeuroImage, 86, 446-460. https://doi.org/10.1016/j.neuroimage.2013.10.027
4. Desikan, R. S., et al. (2006). An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest. Neuroimage, 31(3), 968-980.

### MNE-Python Acknowledgment

AutoClean EEG2Source is built upon the MNE-Python ecosystem and utilizes many of its powerful source localization algorithms. We are deeply grateful to the MNE development team and the broader MNE community for creating and maintaining this essential neuroimaging toolkit. Please visit [MNE-Python](https://mne.tools/stable/index.html) for more information about this incredible framework.

## Building and Publishing

### Build the package

```bash
python -m build
```

### Upload to TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

### Install from TestPyPI

```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps autocleaneeg-eeg2source
```

## License

MIT License
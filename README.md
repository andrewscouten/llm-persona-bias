# Analyzing Social Bias in LLM-Generated Personas through Embedding Space Visualization

[![Python 14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive study examining biases in Large Language Models (LLMs) through analysis of persona-based text generations and embeddings. This project analyzes pre-generated data from the [markedpersonas](https://github.com/myracheng/markedpersonas) project to investigate how demographic attributes influence language model outputs and explore bias patterns.

This study was done as part of the course work in the class CS7313 at Texas State University, taught by Dr. Vangelis Metsis, in the Fall of 2025.

## Authors

- **Andrew Scouten**, this repository
- **Tanha Tahseen**, [Marked-Personas-Replication-Extension](https://github.com/tanhatahseen/Marked-Personas-Replication-Extension)

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone Repository](#clone-repository)
  - [Install Dependencies](#install-dependencies)
- [Development Environment](#development-environment)
- [License](#license)

## Overview

This project analyzes biases in Large Language Models using pre-generated persona data from the [markedpersonas](https://github.com/myracheng/markedpersonas) project. The analysis focuses on:

1. **Embedding Analysis**: Analyzing embeddings from persona-based text generations to identify clustering patterns and biases
2. **Sentiment Analysis**: Examining sentiment variations across different demographic personas
3. **Word Importance**: Identifying key linguistic features associated with demographic attributes
4. **Visualization**: Creating comprehensive visualizations of bias patterns and embedding spaces

The study investigates how demographic characteristics (age, gender, ethnicity, etc.) correlate with language model outputs and how these biases can be quantified and visualized. For further information, see the [Project Proposal](./docs/CS%207313%20–%20Project%20Proposal.pdf) and [Project Requirements](./docs/Project%20Requirements.md).

## Project Structure

```
.
├── docs/               # Project documentation and documents
├── markedpersonas/     # Git submodule with pre-generated persona data
│   ├── data/           # Pre-generated datasets from the markedpersonas project
├── notebooks/          # Project notebooks
├── src/cs7313/         # Source code
├── pyproject.toml      # Project dependencies
└── README.md
```

## Installation

### Prerequisites

- **Python 3.14**
- **Ubuntu 24.04.3 LTS** (or compatible Linux distribution)
- **PyTorch-compatible hardware** (CPU, CUDA, or ROCm)

#### Windows Users (WSL)

For Windows users, we recommend using WSL 2:

```sh
wsl --install -d Ubuntu-24.04
```

#### AMD GPU Setup (ROCm)

If using an AMD GPU, install ROCm support:

```sh
# Install AMD unified driver package
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.4.2.1/ubuntu/noble/amdgpu-install_6.4.60402-1_all.deb
sudo apt install ./amdgpu-install_6.4.60402-1_all.deb

# Install ROCm
amdgpu-install -y --usecase=wsl,rocm --no-dkms

# Verify installation
rocminfo
```

### Clone Repository

```sh
git clone --recurse-submodules https://github.com/andrewscouten/CS7313-Group-Project.git
cd CS7313-Group-Project
```

### Install Dependencies

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management. Choose the appropriate installation based on your hardware:

##### CPU Only
```sh
uv sync --extra cpu
```

##### CUDA 12.8
```sh
uv sync --extra cu128
```

##### CUDA 13.0
```sh
uv sync --extra cu130
```

##### ROCm 6.4
```sh
uv sync --extra rocm
```

**ROCm Post-Installation Troubleshooting**

If PyTorch does not detect your GPU, try the following:
```sh
location=`uv pip show torch | grep Location | awk -F ": " '{print $2}'`
rm ${location}/torch/lib/libhsa-runtime64.so*
cp /opt/rocm/lib/libhsa-runtime64.so ${location}/torch/lib/libhsa-runtime64.so
```

#### Install Project
```sh
uv pip install -e .
```

## Development Environment

### Tested Configuration

- **CPU**: AMD Ryzen 7 7800X3D 8-Core Processor
- **GPU**: AMD Radeon RX 7800 XT (16 GB)
- **RAM**: 32 GB
- **OS**: Windows 11 with WSL 2.6.1.0 running Ubuntu 24.04.3 LTS
- **IDE**: VSCode with WSL, Python, and Jupyter extensions

## License

This project is licensed under the MIT license terms specified in the [LICENSE](LICENSE) file.

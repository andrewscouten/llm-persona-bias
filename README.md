# CS7313 Group Project

**By:** *Tanha Tahseen* and *Andrew Scouten*

## Development Environment

### Andrew's Environment

    PC Specifications:
    CPU: AMD Ryzen 7 7800X3D 8-Core Processor
    GPU: AMD Radeon RX 7800 XT, 16 GB
    RAM: 32 GB

    OS:
    Windows 11
    WSL 2.6.1.0 running Ubuntu 24.04.3 LTS

    VSCode
    Extensions: WSL, Python, Jupyter

### Operating System

This project uses **Ubuntu 24.04.3 LTS**. 

#### WSL 

For a computer running Windows, you can utilize WSL by running the command:
```sh
wsl --install -d Ubuntu-24.04
```

##### AMD GPU Compatibility

If you have an AMD GPU and wish to utilize it, you will need to run the following in WSL (taken from [here](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-radeon.html)):

###### 1. Install AMD unified driver package repositories and installer script
```sh
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.4.2.1/ubuntu/noble/amdgpu-install_6.4.60402-1_all.deb
sudo apt install ./amdgpu-install_6.4.60402-1_all.deb
```

###### 2. Install AMD unified kernel-mode GPU driver, ROCm, and graphics

```sh
amdgpu-install -y --usecase=wsl,rocm --no-dkms
```

You can see other use cases with:
```sh
sudo amdgpu-install --list-usecase
```

###### 3. Post-install verification check
```sh
rocminfo
```

You should see your GPU listed:

    [...]
    *******
    Agent 2
    *******
    Name:                    gfx1100
    Marketing Name:          Radeon RX 7900 XTX
    Vendor Name:             AMD
    [...]
    [...]


### Clone the Repository
```sh
git clone https://github.com/andrewscouten/CS7313-Group-Project.git
cd CS7313-Group-Project
```

### Download Dependencies

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) as a dependency manager. To use uv to install dependencies, the command depends on your PyTorch version requirements listed below.

#### CPU
```sh
uv sync --extra cpu
```

#### CUDA 12.8
```sh
uv sync --extra cu128
```

#### CUDA 13.0
```sh
uv sync --extra cu130
```

#### ROCm 6.4
```sh
uv sync --extra rocm
```

To get PyTorch to use my GPU on ROCm 6.4, I had to do the following (taken from [here](https://www.reddit.com/r/ROCm/comments/1ep4cru/rocm_613_complete_install_instructions_from_wsl/)):

```sh
location=`uv pip show torch | grep Location | awk -F ": " '{print $2}'`
rm ${location}/torch/lib/libhsa-runtime64.so*
cp /opt/rocm/lib/libhsa-runtime64.so ${location}/torch/lib/libhsa-runtime64.so
```


<!-- 
```sh
```
-->
# HGL

## Installation

### 1. Installing CUDA
+ Install CUDA 11.x and compatible drivers from https://developer.nvidia.com/cuda-downloads

### 2. Installing Conda
+ Install Anaconda or Miniconda from https://docs.conda.io

### 3. Creating Environment
+ Create a new clean environment with:
```bash
conda create -n hgl-env python=3.8 && source activate hgl-env
```

### 4. Installing Dependencies
+ [Recommended] Minimal installation requires `pytorch=1.10.2` and `dgl-cuda11.3=0.7.2`, 
```bash
conda install pytorch=1.10.2 torchvision torchaudio torchtext cudatoolkit=11.3 -c pytorch -c conda-forge
conda install dgl-cuda11.3=0.7.2 -c dglteam -c conda-forge
conda install pytest -c conda-forge
pip3 install rdflib
```

+ [Not required] You can also install `pyg=2.0.1` at the same time, however, this may cause conflicts on certain anaconda versions.
```bash
conda install pyg=2.0.1 dgl-cuda11.3=0.7.2 -c pyg -c dglteam -c conda-forge
```

### 5. Installing HGL-proto
+ First, download the repo
```bash
git clone https://github.com/ytgui/HGL-proto.git && cd HGL-proto
```

+ Second, build and install custom dependency
```
python3 setup.py install
```


## 
(Note that this command will build and install a python package named `graph_ext` in your enviroment, which includes all the necessary computation kernels. So that )


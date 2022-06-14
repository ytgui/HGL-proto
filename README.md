# HGL

## Installation
> Please feel free to submit issues if you find any problem during installation.

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

### 6. Cleanup
+ Delete the `hgl-env` from conda.
+ After evaluation, downloaded datasets are saved at `$HOME/.dgl/` 

## Evaluation
> Although this prototype implementation performs excellent performance as the paper illustrated, it is not sufficient for production usage. Please consider to re-implement HGL on top of TVM/MLIR/torch.jit/torch.fx (a better official implementation for production is comming soon).

### 1. Folder Structure
+ `setup.py`: After your `install` command, it installs a python package named `graph_ext` in your enviroment, which includes all the necessary (Het)GNN kernels. So that pure python implemented `HGL-proto` can be decoupled from C/C++/CUDA codes.
+ `src/`: Folder contains C/C++/CUDA codes.
+ `hgl/`: Folder contains pure python prototype implementation of HGL.
+ `test/`: Folder contains all the files for evaluating.

### 2. Known Issues and Limitations
+ `tensor-float-32` feature on ampere GPUs (e.g., RTX30) may affect accuracy, if you add new test cases, please do not forget:
```python
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
```

### 3. Correctness of HGL-proto
> Both forward results and backward gradients are checked carefully.
+ First, check CUDA kernel results multiple times, sparse matrix computations are compared with dense matrix GEMMs, absolute tolerance is set to 1e-3.
```bash
PYTHONPATH=. python3 test/test_kernel.py
```
+ Second, check correctness of GNN layer, make sure HGL doesn't miss important computation or give wrong result.
```
PYTHONPATH=. python3 test/test_allclose.py
```

+ Third, minimal homogeneous training (GAT) example, make sure the convergence of HGL.
```
PYTHONPATH=. python3 test/test_homo.py
```

+ Fourth, minimal heterogeneous training (R-GAT) example, make sure the convergence of HGL.
```
PYTHONPATH=. python3 test/test_hetero.py
```

+ Fifth, stitching, the most complex feature of HGL, is ensured to be correct by comparing model outputs when it is enabled and disabled.
```
PYTHONPATH=. python3 test/test_stitch.py
```


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

+ [Not required] You can also install `pyg=2.0.1` at the same time, however, this may cause conflicts on certain anaconda, gcc, and CUDA versions.
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
+ After evaluation, downloaded datasets are saved at `$HOME/.dgl/`, you can remove this folder if it is not needed further. 

## Evaluation
> Although this prototype implementation performs excellent performance as the paper illustrated, it is not sufficient for production usage. Please consider to re-implement HGL on top of TVM/MLIR/torch.jit/torch.fx (a better official implementation with production stability is comming soon).

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

+ First, check CUDA kernel results multiple times, where sparse matrix computations are compared with dense matrix GEMMs, absolute tolerance is set to 1e-3.
```bash
PYTHONPATH=. python3 test/test_kernel.py
```

+ Second, check correctness of GNN layer, it makes sure HGL doesn't miss important computation or give wrong result.
```bash
PYTHONPATH=. python3 test/test_allclose.py
```

+ Third, minimal homogeneous training example, the case makes sure the convergence of GAT model.
```bash
PYTHONPATH=. python3 test/test_homo.py
```

+ Fourth, minimal heterogeneous training example, the case makes sure the convergence of R-GAT model.
```bash
PYTHONPATH=. python3 test/test_hetero.py
```

+ Fifth, stitching, the most complex feature of HGL, is ensured to be correct by comparing model outputs when it is enabled and disabled.
```bash
PYTHONPATH=. python3 test/test_stitch.py
```

### 4. Result Reproduction
> Some minor experiments may be removed temporarily during major revision phase.

+ First, download and list all the datasets
```bash
PYTHONPATH=. python3 test/bench_macro.py --info
```
+ Expected output:
```bash
----------[aifb-hetero]----------
n_nodes: 7262
node-types: 7
meta-paths: 104
n_rows: 216138
n_edges: 48810
avg_degrees: 4.030543059612123
----------[mutag-hetero]----------
n_nodes: 27163
node-types: 5
meta-paths: 50
n_rows: 281757
n_edges: 148100
avg_degrees: 0.5861625145724975
...
```


+ Second, performance comparison with baseline in terms of throughput and memory consumption

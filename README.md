# SageIR
## Installation
```bash
conda create -n your_env_name python=3.8
conda install pytorch=1.10.2 torchvision torchaudio torchtext cudatoolkit=11.3 -c pytorch -c conda-forge
conda install pyg=2.0.1 dgl-cuda11.3=0.7.2 -c pyg -c dglteam -c conda-forge
conda install pytest -c conda-forge
pip3 install rdflib
```

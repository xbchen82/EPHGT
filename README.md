# Edge-Enhanced Heterogeneous Graph Transformer with Priority-Based Feature Aggregation for Multi-Agent Trajectory Prediction

This repository contains the official implementation of the paper **"[Edge-Enhanced Heterogeneous Graph Transformer with Priority-Based Feature Aggregation for Multi-Agent Trajectory Prediction](https://ieeexplore.ieee.org/document/10807107)"**. The code demonstrates how to leverage edge-enhanced heterogeneous interaction modeling and priority-based feature aggregation for accurate multi-modal trajectory prediction in multi-agent environments.


### Highlights
- Captures interaction heterogeneity among agents and edge attributes.
- Introduces priority mechanisms for decoding and trajectory generation.


## Getting Started

### Prerequisites

Ensure the following dependencies are installed:
- dependencies listed in `requirements.txt`.

Install all dependencies using:
```bash
pip install -r requirements.txt
```

### Train & Test
Use the *.sh scripts. For example, we can train and test EPHGT on sdd dataset by:
```bash
./train_sdd.sh && ./test_sdd.sh
```
### Citation
If you find this work useful in your research, please consider cite:
```
@article{zhou2024ephgt,
  title={Edge-Enhanced Heterogeneous Graph Transformer With Priority-Based Feature Aggregation for Multi-Agent Trajectory Prediction}, 
  author={Zhou, Xiangzheng and Chen, Xiaobo and Yang, Jian},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  year={2024},
  volume={},
  pages={},
  year={2024},
}
```
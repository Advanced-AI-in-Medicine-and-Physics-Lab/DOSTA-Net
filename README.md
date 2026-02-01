
---

# DOSTA-Net: Domain-Shuffle Temporal Attention Network for Vessel Extraction in X-Ray Coronary Angiography Using Synthetic Data

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Artery extraction from X-ray coronary angiography (XCA) images is essential for accurate diagnosis and treatment of coronary artery diseases. This project introduces **DOSTA-Net**, a deep learning framework that leverages synthetic temporal XCA data for training without requiring manual annotations.

| **[Paper Link](https://ieeexplore.ieee.org/document/11369450)** |
**[Pretrained Model Weights](https://drive.google.com/file/d/1ORcWla7-Ca-b07PasN7dhPU-PVGjxXwF/view?usp=sharing)** |


---

## Project Structure

```
.
├── datasets                 # Raw data and external dataset files
├── loss/                    # Custom loss functions and criterion definitions
├── models/                  # Network architectures 
├── util/                    # Utility functions 
├── LICENSE                  # Project license
├── README.md                # Project documentation
├── config.json              # Configuration file for training/inference hyperparameters
├── dataset.py               # Dataset loading and preprocessing logic
├── inference.py             # Script for running model inference
└── requirements.txt         # List of required Python dependencies
```

---

## Installation

```bash
git clone https://github.com/JinkuiH/DOSTA-Net.git
cd DOSTA-Net
conda create -n dostanet python=3.9
conda activate dostanet
pip install -r requirements.txt
```

---

## Inference

Before running inference, please download the pretrained model weights and place them in the weights/ folder. 

You can download our pretrained model from the [Releases](https://drive.google.com/file/d/1ORcWla7-Ca-b07PasN7dhPU-PVGjxXwF/view?usp=sharing) page.

To run inference using pretrained weights:

```bash
python inference.py
```

Results will be saved in the directory specified in the outputs file.

---

## Training

To train the model using synthetic and pseudo-labeled data:

```bash
python training_ours.py
```

Make sure `config.json` is properly set (dataset paths, hyperparameters, etc.).

---


## Citation

If you find this work helpful, please cite:

```bibtex
@article{hao2025dosta,
  title={DOSTA-Net: Domain-Shuffle Temporal Attention Network for Vessel Extraction in X-Ray Coronary Angiography Using Synthetic Data},
  author={Hao, Jinkui and others},
  journal={TBD},
  year={2025}
}
```




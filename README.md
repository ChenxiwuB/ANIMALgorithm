
# ANIMALgorithm
# Part 1: CNNs.ipynb
# Facial Keypoint Detection with JAX & Flax

This code implements a convolutional neural network in JAX/Flax for detecting 68 facial landmarks on 224×224 grayscale images. It includes data-loading utilities, training/evaluation loops, and prediction scripts.

---

## Prerequisites

- **Python** 3.8 or 3.9  
- **CUDA** 11.x toolkit (and appropriate NVIDIA drivers) if you want GPU acceleration  
- **NVIDIA GPU** (e.g. RTX 3090) recommended for fast training, but CPU only will also work  

---

## Create a Compatible Environment

### Using Conda (recommended)

```bash
# 1. Create environment with Python
conda create --name keypoint-cnn python=3.9
conda activate keypoint-cnn

# 2. Install core libraries
conda install -c conda-forge \
    jax jaxlib cudatoolkit=11.1 \
    flax optax \
    numpy scipy scikit-learn \
    opencv-python-headless \
    matplotlib

# 3. (Optional) Install CPU-only JAX if you do not have CUDA
#    conda install -c conda-forge jax jaxlib

# Part 2: Animal algorithm
To run:  
# python scripts/reader/py --idx __
python run.py --idx <image-index>
   - outputs 4-letter code + matching animal(s).


Keypoints labels template:
![alt text](image.png)

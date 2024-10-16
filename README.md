
# Art Generation Using GANs

This project utilizes **Generative Adversarial Networks (GANs)** to generate artistic portraits based on a dataset of images. The model consists of a **Generator** that creates images and a **Discriminator** that distinguishes between real and fake images. Over time, the Generator improves to produce more realistic outputs.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Optimizations](#optimizations)
- [Results](#results)
- [References](#references)

---

## Project Overview

This project uses **PyTorch** to implement a GAN that generates art in the style of portraits. It leverages a dataset of images to train the model to learn artistic features and produce new, realistic images from random noise vectors.

## Dataset

- **Location**: The dataset is stored in the directory `D:/Art-Generation-Using-GANs/Portraits/all_imgs`.
- **Structure**: The images must be placed inside a subfolder for compatibility with PyTorch's `ImageFolder` format.
- **Supported Formats**: The images should be in `.jpg`, `.jpeg`, `.png`, or other standard formats.

Example folder structure:

```
D:/Art-Generation-Using-GANs/Portraits/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Model Architecture

### Generator
The Generator takes in random noise vectors and generates images by progressively applying transposed convolution layers.

### Discriminator
The Discriminator takes in real or fake images and classifies them as real or fake. It consists of convolution layers with LeakyReLU activations and uses a Sigmoid function to output the probability of the image being real.

## Requirements

To run this project, you will need to install the following Python packages:

```bash
pip install torch torchvision
```

Additional requirements:

- **Python 3.x**
- **CUDA** (for GPU acceleration)
- **Mixed Precision** training with `torch.cuda.amp`

## How to Run

1. **Clone the repository**:
   ```bash
    git clone https://github.com/STiFLeR7/Art-Generation-Using-GANs.git
   cd Art-Generation-Using-GANs
   ```

2. **Prepare your dataset**:
   Ensure the dataset is in the correct folder structure as described above.

3. **Run the training script**:
   Execute the following command to start training:
   ```bash
   python main.py
   ```

   The script will train the model and save generated images after each epoch in the `output/` directory.

4. **Monitor training**:
   The script will print the loss for both the Generator and Discriminator at each step.

## Optimizations

The project includes the following optimizations to reduce training time:

- **Mixed Precision Training**: Using `torch.cuda.amp` for faster computations with lower precision.
- **AdamW Optimizer**: For potentially faster convergence.
- **Batch Size**: Increased based on GPU memory to speed up training.
- **Early Stopping**: The model can terminate training early if the losses stabilize.

## Results

Generated images are saved in the `output/` folder after each epoch. Example outputs can be viewed to track the model's progress over time.

## References

- **PyTorch Documentation**: [PyTorch Official Website](https://pytorch.org/docs/stable/index.html)
- **GAN Paper**: [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)

---

# Deep Learning Image Classification with GPU Acceleration

## Project Summary

This project focuses on developing a deep learning image classification model and optimizing its training using GPU parallelism. We compare the performance between CPU and GPU training, showcasing the significant speedup GPU provides. The model utilizes the CIFAR-10 dataset and implements parts of the neural network using GPU programming techniques to accelerate operations like matrix multiplications and convolutions.

## Key Features

- **GPU Acceleration:** Leverages GPU capabilities to significantly reduce training time compared to CPU.
- **CIFAR-10 Dataset:** Uses the CIFAR-10 dataset for image classification.
- **CNN Model:** Employs a Convolutional Neural Network (CNN) for image classification.
- **Performance Comparison:** Includes a detailed comparison of training performance on CPU and GPU.
- **PyTorch Framework:** Built using the PyTorch deep learning framework.

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Torchaudio
- PyCUDA
- CUDA Toolkit (if using GPU)

## Installation

1. Install the required libraries using pip:

# CycleGAN-Implementation-for-Image-To-Image-Translation

 🚀 In this project, we've implemented CycleGAN for image-to-image translation using PyTorch. CycleGAN is a powerful deep learning model that learns to translate images from one domain to another without paired examples.

## Overview

CycleGAN is a type of generative adversarial network (GAN) that consists of two generators and two discriminators. It learns to map images from one domain (e.g., horses) to another domain (e.g., zebras) in an unsupervised manner. This means that it doesn't require paired examples during training.

## Tech Stack Used

- **Python**: Programming language used for implementation.
- **PyTorch**: Deep learning framework used for building and training the CycleGAN model.
- **Pillow**: Python Imaging Library used for image processing tasks.
- **NumPy**: Library used for numerical computations.
- **torchvision**: Library providing datasets, transforms, and common image processing utilities for PyTorch.
- **GitHub**: Version control system for collaboration and code management.

## Architecture

The architecture of the CycleGAN model consists of the following components:

- **Generator (G_AB and G_BA)**: Converts images from one domain to another. It consists of Residual Blocks for effective feature extraction and upsampling/downsampling layers for image transformation.
- **Discriminator (D_A and D_B)**: Discriminates between real and fake images from each domain. It is a convolutional neural network used for adversarial training.
- **Loss Functions**: Includes adversarial loss, cycle-consistency loss, and identity loss, which are used to train the generators and discriminators.
- **Replay Buffer**: Used for storing and retrieving previously generated samples during training.

## Files in the Project

- **cyclegan.py**: Main script containing the training loop and model configurations.
- **datasets.py**: Custom dataset class for loading and preprocessing image data.
- **models.py**: Definitions of Generator and Discriminator models.
- **utils.py**: Utility functions including weight initialization and learning rate scheduling.
- **requirements.txt**: List of Python dependencies required to run the project.

## How to Use

1. Ensure that you have the required dependencies installed. You can use the `requirements.txt` file to install them.
2. Run the `cyclegan.py` script to start training the CycleGAN model.
3. Monitor the training progress and generated images in the `images/` directory.
4. Adjust hyperparameters and experiment with different configurations as needed.

## Let's Connect!

- **Kaggle**: [Vidhi Waghela on Kaggle](https://www.kaggle.com/vidhikishorwaghela)
- **LinkedIn**: [Vidhi Waghela](https://www.linkedin.com/in/vidhi-waghela-434663198/)
- **GitHub**: [Vidhi Waghela on GitHub](https://github.com/Vidhi1290)

Feel free to explore the code, contribute, and reach out for collaborations or discussions!
Thank you for exploring Project Pro! 🎉

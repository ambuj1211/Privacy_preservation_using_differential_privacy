# Privacy Preservation using Differential Privacy in MNIST DATASET

This repository contains an implementation of differentially private (DP) deep learning for image classification on the MNIST dataset. It demonstrates how to train neural networks with strong privacy guarantees while maintaining high utility.

## Overview

Differential privacy provides mathematical guarantees that models trained on sensitive data will not reveal information about individual training examples. This project explores:

1. Implementation of Differentially Private Stochastic Gradient Descent (DP-SGD)
2. Privacy-utility trade-offs for different privacy settings
3. Visualization of model performance and privacy guarantees
4. Analysis of adjacent datasets to demonstrate privacy protection

## Features

- Custom implementation of DP-SGD with gradient clipping and noise addition
- Privacy accountant to track cumulative privacy loss (ε)
- Differentially private dimensionality reduction using PCA
- Privacy budget exploration across multiple noise levels
- Membership inference attack demonstration
- Adjacent dataset analysis to empirically verify privacy guarantees

## Components

### Core Classes

- `PrivacyAccountant`: Tracks privacy budget spending across iterations
- `GradientSanitizer`: Implements gradient clipping and noise addition
- `DPSGD_Optimizer`: Handles privacy-preserving optimization
- `AdjacentDatasetAnalyzer`: Tests model behavior on adjacent datasets (differing by one record)

### Key Functions

- `train_model()`: Trains neural networks with differential privacy
- `pca_with_privacy()`: Performs dimensionality reduction with privacy guarantees
- `privacy_budget_exploration()`: Evaluates model performance across privacy settings
- `adjacent_dataset_analysis()`: Tests privacy protection empirically
- `membership_inference_demonstration()`: Shows reduced membership inference risk

## Privacy-Utility Trade-offs

The implementation demonstrates how different noise multipliers affect:

- Final model accuracy
- Privacy budget (ε)
- Susceptibility to membership inference
- Stability across adjacent datasets

Higher noise multipliers provide stronger privacy guarantees at the cost of reduced accuracy, while lower noise settings prioritize utility over privacy.

## Usage

The main workflow:

1. Load and preprocess MNIST data
2. Apply differentially private PCA for dimensionality reduction
3. Train neural network with DP-SGD
4. Evaluate model performance and privacy guarantees
5. Analyze adjacent datasets to verify privacy protection

## Visualization

The project includes comprehensive visualization:

- Training curves (loss, accuracy, privacy budget)
- Confusion matrix for model predictions
- PCA reconstructions of original images
- Per-digit accuracy analysis
- Privacy-utility trade-off curves
- Adjacent dataset prediction differences

## Privacy Analysis

Key privacy analyses include:

1. **Membership Inference**: Demonstrates reduced information leakage about training set membership
2. **Adjacent Dataset Analysis**: Shows that removing a single example has minimal impact on model predictions
3. **Privacy Budget Exploration**: Visualizes the relationship between noise, privacy budget, and utility

## Results

With a noise multiplier of 1.1 and appropriate clipping, the model achieves:
- Good classification accuracy on MNIST
- Strong differential privacy guarantees (ε < 10)
- Significant protection against membership inference attacks
- Minimal prediction differences when training examples are removed

## Requirements

- Python 3.10
- jupyter notebook 7.4.0
- TensorFlow 
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

## Additional Resources

To learn more about differential privacy in deep learning:

- [Deep Learning with Differential Privacy (Abadi et al.)](https://arxiv.org/abs/1607.00133)
- [The Algorithmic Foundations of Differential Privacy (Dwork & Roth)](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- [Privacy Accounting and the Moments Accountant](https://arxiv.org/abs/1607.00133)

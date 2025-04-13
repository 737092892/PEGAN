# Introduction

This project introduces a Generative Adversarial Network (GAN) designed specifically for augmenting functional Magnetic Resonance Imaging (fMRI) data. The goal of this tool is to enhance the quality and quantity of fMRI datasets, which can be particularly useful for improving the performance of downstream tasks such as classification, regression, and anomaly detection.

# Methodology
The proposed GAN architecture, named PE-GAN (Patch-Entirety GAN), is designed to address the challenges of generating realistic and high-quality fMRI data. PE-GAN leverages a convolutional architecture combined with two-stage attention mechanisms and symmetry constraints to ensure that the generated data retains both local and global structural consistency. This design is validated through systematic experiments comparing it with baseline GAN variants, including a fully-connected GAN and a convolutional GAN without attention or symmetry constraints.

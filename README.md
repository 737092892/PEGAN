# Introduction
This project introduces a Generative Adversarial Network (GAN) designed specifically for augmenting functional Magnetic Resonance Imaging (fMRI) data. The goal of this tool is to enhance the quality and quantity of fMRI datasets, which can be particularly useful for improving the performance of downstream tasks such as classification, regression, and anomaly detection.
# Methodology
The proposed GAN architecture, named PE-GAN (Patch-Entirety GAN), is designed to address the challenges of generating realistic and high-quality fMRI data. PE-GAN leverages a convolutional architecture combined with two-stage attention mechanisms and symmetry constraints to ensure that the generated data retains both local and global structural consistency. This design is validated through systematic experiments comparing it with baseline GAN variants, including a fully-connected GAN and a convolutional GAN without attention or symmetry constraints.
# Dataset
The ADHD-200 dataset is used for training and evaluation. This dataset consists of multi-site fMRI data from NYU, PKU, and NI, providing a diverse and representative sample of functional brain activity. The cross-site nature of the dataset allows us to evaluate the generalization capability of the proposed method across different populations and acquisition protocols.
# Key Contributions
1.Enhanced Data Augmentation: PE-GAN demonstrates superior performance in generating realistic fMRI data to compared traditional GAN architectures, as validated through classification tasks on NYU, PKU, and NI datasets.
2.Attention Mechanism Integration: The introduction of Patch-Entirety self-attention improves the quality and relevance of generated data, highlighting the importance of attention mechanisms in GAN-based data augmentation.
3.Cross-Site Generalization: By averaging performance across sites, the proposed method reduces the impact of site-specific characteristics, leading to more robust and generalizable models.
Practical Applications: The augmented data can be used to improve downstream tasks such as ADHD classification, regression analysis, and anomaly detection, addressing challenges associated with limited or imbalanced fMRI datasets.
Results and Impact

Experiments show that PE-GAN outperforms baseline models in terms of classification accuracy and data quality metrics. The augmented data not only increases the quantity of training samples but also enhances the robustness of machine learning models trained on fMRI data. This approach is particularly valuable in scenarios where acquiring additional real fMRI data is costly or impractical.
By providing a flexible and effective tool for fMRI data augmentation, this project aims to advance research in neuroimaging analysis and support the development of more accurate and reliable diagnostic tools for conditions such as ADHD.

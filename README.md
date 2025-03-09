# AI vs Human Image Classifier

## Overview
This project tackles the crucial challenge of distinguishing between AI-generated and human-created images—a task of growing importance as generative AI becomes increasingly sophisticated in creating realistic content. Using advanced computer vision and deep learning techniques, this notebook provides a comprehensive solution that achieves high accuracy in classifying images as either AI-generated or human-created.

The ability to detect AI-generated content has significant implications for media authenticity, digital forensics, and combating misinformation. This project contributes to the broader effort of maintaining trust in digital media by providing reliable detection methods.

## Dataset
The dataset is provided by Shutterstock and DeepMedia for the "Can You Tell the Difference?" competition:

- **Training set**: 80,000 images
- **Test set**: 5,500 images
- **Classes**: Human-created (0) and AI-generated (1)

The authentic images are sourced from Shutterstock's extensive collection, with approximately one-third featuring humans. The AI-generated counterparts were created by DeepMedia using state-of-the-art generative models. This pairing ensures a direct comparison between real and AI-generated content, creating a robust foundation for training detection systems.

## Technical Environment

### Hardware
This project was developed and tested using:
- NVIDIA Tesla P100 GPU (16GB VRAM)
- CUDA 11.2
- 25GB+ RAM recommended

### Software
The solution is implemented as a single Jupyter notebook running on:
- Python 3.8+
- PyTorch 1.9+
- CUDA-enabled environment (Kaggle/Colab)

## Notebook Contents

The notebook follows a structured approach to solving this classification problem:

1. **Setup and Configuration**
   - Library imports and version checks
   - GPU configuration and memory optimization
   - Hyperparameter settings
   - Reproducibility setup (fixed seeds)

2. **Data Exploration**
   - Dataset loading and statistical analysis
   - Class distribution visualization
   - Sample image rendering from both classes
   - Detailed analysis of image characteristics:
     - Size distributions (width/height) comparison between AI and human images
     - Brightness patterns and intensity profiles
     - Color distribution analysis

3. **Data Preprocessing**
   - Advanced image transformations:
     - Resize with bicubic interpolation
     - Random cropping and flips
     - Color jittering and normalization
   - Custom dataset implementation with efficient loading
   - Stratified train-validation split (95%/5%)

4. **Model Development**
   - Implementation of RegNet Y-32GF architecture
   - Custom model head for binary classification
   - Training and validation pipeline development
   - Optimization strategy with Adam optimizer
   - Learning rate scheduling with step decay

5. **Model Training**
   - Comprehensive training loop with:
     - Mixed precision training for GPU optimization
     - Batch progress visualization
     - Epoch-level metrics tracking
     - Model checkpointing based on validation accuracy
   - Loss and accuracy visualizations

6. **Inference and Submission**
   - Test-time augmentation strategy
   - Prediction visualization with sample images
   - Confidence scoring
   - Submission file generation in competition format

## Performance Optimization
- **Memory Optimization**: Implements efficient data loading with proper batch sizing for P100 GPU
- **CPU Parallelism**: Leverages multiprocessing for data preprocessing
- **CUDA Optimization**: Configured for optimal GPU utilization

## Results
The RegNet Y-32GF model demonstrates strong performance in distinguishing between AI-generated and human-created images:
- Training accuracy: 0.998
- Validation accuracy: 0.9988
- F1 Score: 0.9987

Performance metrics are tracked and visualized throughout the training process, with detailed classification reports available in the notebook.

## How to Use

### Requirements
The notebook requires the following libraries:

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0
pandas>=1.3.0
opencv-python>=4.5.3
pillow>=8.3.1
matplotlib>=3.4.3
seaborn>=0.11.2
scikit-learn>=0.24.2
tqdm>=4.62.2
albumentations>=1.0.3
timm>=0.5.4
```

A requirements.txt file is included in the repository.

### Running the Notebook
1. Upload the notebook to a platform with P100 GPU support or equivalent (Kaggle/Colab recommended)
2. Install the required dependencies:
   ```
   !pip install -r requirements.txt
   ```
3. Ensure access to the competition dataset
4. Run all cells sequentially
5. For custom inference, modify the test data path in the configuration section

### Dataset Access
The complete dataset is available at: https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset

## Competition
This project was developed for the "Can You Tell the Difference?" competition hosted by Women in AI. The competition challenges participants to design machine learning models that can accurately classify images as AI-generated or human-created, while ensuring fairness and robust performance across diverse data.

More information can be found at: https://www.womeninai.co/kagglechallenge2025

## Future Improvements
- Model ensemble strategies combining multiple architectures
- Additional data augmentation techniques specific to AI-generated content
- Explainable AI methods to highlight distinguishing features
- Adversarial testing to improve robustness against evolving generation methods


## Acknowledgements
- Shutterstock and DeepMedia for providing the comprehensive dataset
- Women in AI for hosting and organizing the competition
- NVIDIA for GPU computing resources

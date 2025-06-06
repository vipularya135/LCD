# Active Learning Experiments

## Problem Statement

Deep learning models including Convolutional Neural Networks and Vision Transformers have achieved state-of-the-art performance on many computer vision tasks such as object classification, detection, segmentation, generation and many more. However, these models are data hungry since they require large amount of training data to learn the huge number of parameters or weights. Especially working with supervised learning tasks, curating a large number of labeled images for model training is an expensive and time consuming task. Active Learning (AL) has been used to address this problem for many years. Existing active learning methods aim at choosing the samples for annotation from a pool of unlabelled set that are either diverse or uncertain. Choosing such samples may hinder the model performance since we are pooling w.r.t. one dimension i.e., either diverse or uncertain. In this paper, we propose a novel hybrid sampling method for pooling both easy and hard samples which are also diverse. To verify the efficacy of the proposed method, experiments are conducted considering high and low confidence samples separately. It is evident from the experimental results that the proposed hybrid sampling method helps the deep learning models to achieve better results. 

## Experiments Performed

We conducted experiments using the following models and methods:

1. **Models:**
   - VGG-16
   - ResNet-18
   - ResNet-50
   - ResNet-56
   - Mobilenet
   - DenseNet-121
   - ViT-Small Transformer
   - Swin Transformer

2. **Datasets:**
   - CIFAR-10
   - CIFAR-100
   - SVHN
   - PascalVOC-2012

3. **Experiment Details:**
   - We train our model initially on 10k samples.
   - Then we use 4 methods to select the next 5k samples to train the model:
     1. Method 1: top 5k samples that are diverse low and diverse high confidence.
     2. Method 2: top 5k samples that are diverse and high confidence.
     3. Method 3: top 5k samples that are diverse and low confidence.
     4. Method 4: top 5k samples that are low and high confidence.
   - We retrain the model with these new samples until there is a limit degradation and then check the accuracy after each iteration during retraining for each of these methods.
   - The order from best to worst accuracy is as follows:
     - Diverse and low confidence
     - Diverse and low + high confidence
     - Low and high confidence
     - Diverse and high confidence

4. **Parameters:**
   - Learning Rate: 0.01
   - Epochs: 50 for both initial training and sampling iterations
   - Loss Function: Cross Entropy Loss
   - Optimizer: Stochastic Gradient Descent (SGD)
   - Scheduler: StepLR

## Results


| Dataset      | Model        | LC + Diverse | HC + Diverse | LC + HC  | LC + HC + Diverse |
| ------------ | ------------ | ------------ | ------------ | -------- | ------------------ |
| **CIFAR-10** | VGG-16       | 90.86%       | 84.13%       | 88.57%   | 89.95%             |
|              | ResNet-18    | 91.96%       | 86.67%       | 90.49%   | 91.02%             |
|              | ResNet-50    | 90.39%       | 80.95%       | 89.08%   | 89.46%             |
|              | ResNet-56    | 91.67%       | 83.00%       | 88.35%   | 88.70%             |
|              | Mobilenet    | 87.68%       | 78.78%       | 85.79%   | 86.13%             |
|              | DenseNet-121 | 92.47%       | 85.18%       | 91.54%   | 91.90%             |
|              | Swin         | 81.68%       | 69.96%       | 77.67%   | 78.64%             |
|              | ViT-Small    | 79.60%       | 68.75%       | 76.01%   | 76.46%             |
| **CIFAR-100**| ResNet-18    | 68.31%       | 57.39%       | 61.77%   | 64.35%             |
|              | ResNet-50    | 66.88%       | 56.97%       | 64.82%   | 65.12%             |
|              | DenseNet-121 | 70.29%       | 60.98%       | 66.57%   | 66.91%             |
|              | Swin         | 54.01%       | 42.94%       | 49.98%   | 49.63%             |
| **SVHN**     | VGG-16       | 94.20%       | 89.60%       | 93.75%   | 93.80%             |
|              | ResNet-18    | 95.67%       | 93.71%       | 95.68%   | 95.71%             |
|              | ResNet-50    | 95.63%       | 94.31%       | 95.63%   | 95.83%             |
|              | ResNet-56    | 94.97%       | 93.07%       | 95.41%   | 95.21%             |
|              | Mobilenet    | 94.34%       | 92.65%       | 94.07%   | 94.38%             |
|              | DenseNet-121 | 96.1%        | 94.66%       | 95.93%   | 95.85%             |


## Conclusion

Our experiments indicate that selecting low confidence and diverse samples generally results in the highest accuracy improvement across various models and datasets. The order of effectiveness from best to worst in our experiments was:
1. Diverse and low confidence
2. Diverse and low + high confidence
3. Low and high confidence
4. Diverse and high confidence



## Getting Started

To get started, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/vipularya135/Active-Learning-Sampling-Techniques
    cd Active-Learning-Sampling-Techniques
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the `main.py` script, by selecting the model and dataset:
    ```bash
    python main.py 
    ```

## File Descriptions

### 1. Function_Definitions.py
This file contains the definitions of various functions used throughout the project. These functions might include data preprocessing, model evaluation, and utility functions to streamline the workflow.

### 2. main.py
This is the main execution script of the project. It coordinates the entire process, from data loading and preprocessing to model training and evaluation. It serves as the entry point to run the project.

### 3. models.py
This file defines the architecture of the machine learning or deep learning models used in the project. It may include custom model classes, layers, and configurations necessary for training and inference.

### 4. requirements.txt
This file lists all the Python dependencies and libraries required to run the project. It ensures that anyone who clones the repository can install the exact versions of the dependencies needed to avoid compatibility issues.

### 5. swin.py
This file implements the Swin Transformer model, a type of Vision Transformer known for its hierarchical design and efficiency in handling computer vision tasks.

### 6. vit-tiny.py
This file implements the Vision Transformer (ViT) model, specifically the tiny variant, which is designed for image classification tasks with a smaller parameter count and faster training times.

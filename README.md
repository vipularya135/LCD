# Active Learning Experiments

## Problem Statement

The goal of our research is to evaluate the performance of different active learning methods on various models using CIFAR-10, CIFAR-100, and SVHN datasets. Active learning aims to improve the efficiency of the learning process by selecting the most informative data points to label, thus reducing the amount of labeled data required while maintaining high accuracy.

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

### CIFAR-10

| Models       | Methods              | Final Accuracy |
| ------------ | -------------------- | -------------- |
| **VGG-16**   | LC + Diverse         | 90.86%         |
|              | HC + Diverse         | 84.13%         |
|              | LC + HC              | 88.57%         |
|              | LC + HC + Diverse    | 89.95%         |
| **ResNet-18**| LC + Diverse         | 91.96%         |
|              | HC + Diverse         | 86.67%         |
|              | LC + HC              | 90.49%         |
|              | LC + HC + Diverse    | 91.02%         |
| **ResNet-50**| LC + Diverse         | 90.39%         |
|              | HC + Diverse         | 80.95%         |
|              | LC + HC              | 89.08%         |
|              | LC + HC + Diverse    | 89.46%         |
| **ResNet-56**| LC + Diverse         | 91.67%         |
|              | HC + Diverse         | 83.00%         |
|              | LC + HC              | 88.35%         |
|              | LC + HC + Diverse    | 88.70%         |
| **Mobilenet**| LC + Diverse         | 87.68%         |
|              | HC + Diverse         | 78.78%         |
|              | LC + HC              | 85.79%         |
|              | LC + HC + Diverse    | 86.13%         |
| **DenseNet-121**| LC + Diverse      | 92.47%         |
|              | HC + Diverse         | 85.18%         |
|              | LC + HC              | 91.54%         |
|              | LC + HC + Diverse    | 91.90%         |
| **Swin**     | LC + Diverse         | 81.68%         |
|              | HC + Diverse         | 69.96%         |
|              | LC + HC              | 77.67%         |
|              | LC + HC + Diverse    | 78.64%         |
| **ViT-Small**| LC + Diverse         | 79.60%         |
|              | HC + Diverse         | 68.75%         |
|              | LC + HC              | 76.01%         |
|              | LC + HC + Diverse    | 76.46%         |

### CIFAR-100

| Models       | Methods              | Final Accuracy |
| ------------ | -------------------- | -------------- |
| **ResNet-18**| LC + Diverse         | 68.31%         |
|              | HC + Diverse         | 57.39%         |
|              | LC + HC              | 61.77%         |
|              | LC + HC + Diverse    | 64.35%         |
| **ResNet-50**| LC + Diverse         | 66.88%         |
|              | HC + Diverse         | 56.97%         |
|              | LC + HC              | 64.82%         |
|              | LC + HC + Diverse    | 65.12%         |
| **DenseNet-121**| LC + Diverse      | 70.29%         |
|              | HC + Diverse         | 60.98%         |
|              | LC + HC              | 66.57%         |
|              | LC + HC + Diverse    | 66.91%         |
| **Swin**     | LC + Diverse         | 54.01%         |
|              | HC + Diverse         | 42.94%         |
|              | LC + HC              | 49.98%         |
|              | LC + HC + Diverse    | 49.63%         |

### SVHN

| Models       | Methods              | Final Accuracy |
| ------------ | -------------------- | -------------- |
| **VGG-16**   | LC + Diverse         | 94.20%         |
|              | HC + Diverse         | 89.60%         |
|              | LC + HC              | 93.75%         |
|              | LC + HC + Diverse    | 93.80%         |
| **ResNet-18**| LC + Diverse         | 95.67%         |
|              | HC + Diverse         | 93.71%         |
|              | LC + HC              | 95.68%         |
|              | LC + HC + Diverse    | 95.71%         |
| **ResNet-50**| LC + Diverse         | 95.63%         |
|              | HC + Diverse         | 94.31%         |
|              | LC + HC              | 95.63%         |
|              | LC + HC + Diverse    | 95.83%         |
| **ResNet-56**| LC + Diverse         | 94.97%         |
|              | HC + Diverse         | 93.07%         |
|              | LC + HC              | 95.41%         |
|              | LC + HC + Diverse    | 95.21%         |
| **Mobilenet**| LC + Diverse         | 94.34%         |
|              | HC + Diverse         | 92.65%         |
|              | LC + HC              | 94.07%         |
|              | LC + HC + Diverse    | 94.38%         |

## Conclusion

Our experiments indicate that selecting low confidence and diverse samples generally results in the highest accuracy improvement across various models and datasets. The order of effectiveness from best to worst in our experiments was:
1. Diverse and low confidence
2. Diverse and low + high confidence
3. Low and high confidence
4. Diverse and high confidence

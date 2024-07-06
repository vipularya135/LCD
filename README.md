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

\begin{table}[h]
\centering
\begin{tabular}{|l|l|r|}
\hline
\multicolumn{3}{|c|}{\textbf{CIFAR-10}} \\
\hline
\textbf{Models} & \textbf{Methods} & \textbf{Final Accuracy} \\
\hline
\multirow{4}{*}{\textbf{VGG-16}} 
  & LC + Diverse                & 90.86\% \\
  & HC + Diverse                & 84.13\% \\
  & LC + HC                     & 88.57\% \\
  & LC + HC + Diverse           & 89.95\% \\
\hline
\multirow{4}{*}{\textbf{ResNet-18}} 
  & LC + Diverse                & 91.96\% \\
  & HC + Diverse                & 86.67\% \\
  & LC + HC                     & 90.49\% \\
  & LC + HC + Diverse           & 91.02\% \\
\hline
\multirow{4}{*}{\textbf{ResNet-50}} 
  & LC + Diverse                & 90.39\% \\
  & HC + Diverse                & 80.95\% \\
  & LC + HC                     & 89.08\% \\
  & LC + HC + Diverse           & 89.46\% \\
\hline
\multirow{4}{*}{\textbf{ResNet-56}} 
  & LC + Diverse                & 91.67\%    \\
  & HC + Diverse                & 83\%    \\
  & LC + HC                     & 88.35\% \\
  & LC + HC + Diverse           & 88.7\%  \\
\hline
\multirow{4}{*}{\textbf{Mobilenet}} 
  & LC + Diverse                & 87.68\% \\
  & HC + Diverse                & 78.78\% \\
  & LC + HC                     & 85.79\% \\
  & LC + HC + Diverse           & 86.13\% \\
\hline
\multirow{4}{*}{\textbf{DenseNet-121}} 
  & LC + Diverse                & 92.47\% \\
  & HC + Diverse                & 85.18\% \\
  & LC + HC                     & 91.54\% \\
  & LC + HC + Diverse           & 91.9\%  \\
\hline
\multirow{4}{*}{\textbf{Swin}} 
  & LC + Diverse                & 81.68\% \\
  & HC + Diverse                & 69.96\% \\
  & LC + HC                     & 77.67\% \\
  & LC + HC + Diverse           & 78.64\% \\
\hline
\multirow{4}{*}{\textbf{ViT-Small}} 
  & LC + Diverse                & 79.6\%  \\
  & HC + Diverse                & 68.75\% \\
  & LC + HC                     & 76.01\% \\
  & LC + HC + Diverse           & 76.46\% \\
\hline
\end{tabular}
\caption{ Accuracy of Experiments on CIFAR-10}
\end{table}

### CIFAR-100

\begin{table}[h]
\centering
\begin{tabular}{|l|l|r|}
\hline
\multicolumn{3}{|c|}{\textbf{CIFAR-100}} \\
\hline
\textbf{Models} & \textbf{Methods} & \textbf{Final Accuracy} \\
\hline
\multirow{4}{*}{\textbf{ResNet-18}} 
  & LC + Diverse                & 68.31\% \\
  & HC + Diverse                & 57.39\% \\
  & LC + HC                     & 61.77\% \\
  & LC + HC + Diverse           & 64.35\% \\
\hline
\multirow{4}{*}{\textbf{ResNet-50}} 
  & LC + Diverse                & 66.88\% \\
  & HC + Diverse                & 56.97\% \\
  & LC + HC                     & 64.82\% \\
  & LC + HC + Diverse           & 65.12\% \\
\hline
\multirow{4}{*}{\textbf{DenseNet-121}} 
  & LC + Diverse                & 70.29\% \\
  & HC + Diverse                & 60.98\% \\
  & LC + HC                     & 66.57\% \\
  & LC + HC + Diverse           & 66.91\%  \\
\hline
\multirow{4}{*}{\textbf{Swin}} 
  & LC + Diverse                & 54.01\% \\
  & HC + Diverse                & 42.94\% \\
  & LC + HC                     & 49.98\% \\
  & LC + HC + Diverse           & 49.63\% \\
\hline
\end{tabular}
\caption{ Accuracy of Experiments on CIFAR-100}
\end{table}

### SVHN

\begin{table}[h]
\centering
\begin{tabular}{|l|l|r|}
\hline
\multicolumn{3}{|c|}{\textbf{SVHN}} \\
\hline
\textbf{Models} & \textbf{Methods} & \textbf{Final Accuracy} \\
\hline
\multirow{4}{*}{\textbf{VGG-16}} 
  & LC + Diverse                & 94.2\% \\
  & HC + Diverse                & 89.6\% \\
  & LC + HC                     & 93.75\% \\
  & LC + HC + Diverse           & 93.8\% \\
\hline
\multirow{4}{*}{\textbf{ResNet-18}} 
  & LC + Diverse                & 95.67\% \\
  & HC + Diverse                & 93.71\% \\
  & LC + HC                     & 95.68\% \\
  & LC + HC + Diverse           & 95.71\% \\
\hline
\multirow{4}{*}{\textbf{ResNet-50}} 
  & LC + Diverse                & 95.63\% \\
  & HC + Diverse                & 94.31\% \\
  & LC + HC                     & 95.63\% \\
  & LC + HC + Diverse           & 95.83\% \\
\hline
\multirow{4}{*}{\textbf{ResNet-56}} 
  & LC + Diverse                & 95.24\% \\
  & HC + Diverse                & 91.5\% \\
  & LC + HC                     & 95.18\% \\
  & LC + HC + Diverse           & 95.42\% \\
\hline
\multirow{4}{*}{\textbf{Mobilenet}} 
  & LC + Diverse                & 94.68\% \\
  & HC + Diverse                & 91.31\% \\
  & LC + HC                     & 94.47\% \\
  & LC + HC + Diverse           & 94.69\% \\
\hline
\multirow{4}{*}{\textbf{DenseNet-121}} 
  & LC + Diverse                & 96.1\%  \\
  & HC + Diverse                & 93.74\% \\
  & LC + HC                     & 95.88\% \\
  & LC + HC + Diverse           & 96.01\% \\
\hline
\multirow{4}{*}{\textbf{Swin}} 
  & LC + Diverse                & 93.58\% \\
  & HC + Diverse                & 86.92\% \\
  & LC + HC                     & 91.88\% \\
  & LC + HC + Diverse           & 92.19\% \\
\hline
\multirow{4}{*}{\textbf{ViT-Small}} 
  & LC + Diverse                & 93.26\% \\
  & HC + Diverse                & 88.97\% \\
  & LC + HC                     & 92.21\% \\
  & LC + HC + Diverse           & 92.46\% \\
\hline
\end{tabular}
\caption{ Accuracy of Experiments on SVHN}
\end{table}

## Conclusion

Our experiments indicate that selecting low confidence and diverse samples generally results in the highest accuracy improvement across various models and datasets. The order of effectiveness from best to worst in our experiments was:
1. Diverse and low confidence
2. Diverse and low + high confidence
3. Low and high confidence
4. Diverse and high confidence

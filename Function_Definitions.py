import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import math
from torchvision.datasets import VOCDetection

# Ensure reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch
from torchvision import datasets, transforms

def prepare_data(dataset_name):
    print('==> Preparing data..')
    
    if dataset_name == 'cifar10':
        # CIFAR-10 Normalization
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    elif dataset_name == 'cifar100':
        # CIFAR-100 Normalization
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    elif dataset_name == 'svhn':
        # SVHN Normalization
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        train_set = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        test_set = datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    elif dataset_name == 'voc2012':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        train_set = VOCDetection(root='./data', year='2012', image_set='train', download=True, transform=transform)
        test_set = VOCDetection(root='./data', year='2012', image_set='val', download=True, transform=transform)

        
    else:
        raise ValueError("Invalid dataset name. Choose from 'cifar10', 'cifar100', 'svhn', or 'voc2012'.")

    torch.manual_seed(42)
    initial_train_set, remainder = torch.utils.data.random_split(train_set, [10000, len(train_set) - 10000])

    print(f"Size of train_set: {len(train_set)}")
    print(f"Size of test_set: {len(test_set)}")

    return initial_train_set, remainder, test_set


        
def train_model(model, train_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
     
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
        
        # Step the scheduler
        scheduler.step()
    


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return accuracy

def calculate_cluster_centers(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers

def get_most_diverse_samples(tsne_results, cluster_centers, num_diverse_samples):
    distances = euclidean_distances(tsne_results, cluster_centers)
    min_distances = np.max(distances, axis=1)
    sorted_indices = np.argsort(min_distances)
    diverse_indices = sorted_indices[:num_diverse_samples]
    return diverse_indices

def extract_embeddings(model, test):
    test_loader = data.DataLoader(test, batch_size=64, shuffle=False)
    model.eval()
    embeddings = []
    targets_list = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            intermediate_features = model(images)
            embeddings.extend(intermediate_features.view(intermediate_features.size(0), -1).tolist())
            targets_list.append(targets)
    return embeddings

def least_confidence_images(model, test_dataset, k=2500):
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    confidences = []
    labels = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().tolist())
            labels.extend(targets.cpu().tolist())
    confidences = torch.tensor(confidences)
    k = min(k, len(confidences))
    _, indices = torch.topk(confidences, k, largest=False)
    return data.Subset(test_dataset, indices), [labels[i] for i in indices]

def high_confidence_images(model, test_dataset, k=2500):
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    confidences = []
    labels = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().tolist())
            labels.extend(targets.cpu().tolist())
    confidences = torch.tensor(confidences)
    k = min(k, len(confidences))
    _, indices = torch.topk(confidences, k, largest=True)
    return data.Subset(test_dataset, indices), [labels[i] for i in indices]
    
def HC_diverse(embed_model, remainder, n=None):
    high_conf_images, high_conf_indices = high_confidence_images(embed_model, remainder, k=len(remainder) if 2*n > len(remainder) else 2*n)
    high_conf_embeddings = extract_embeddings(embed_model, high_conf_images)
    high_conf_embeddings = np.array([np.array(e) for e in high_conf_embeddings])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(high_conf_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_indices = get_most_diverse_samples(tsne_results, cluster_centers, n)
    diverse_images = data.Subset(high_conf_images, diverse_indices)
    return diverse_images, [high_conf_indices[i] for i in diverse_indices]

def LC_diverse(embed_model, remainder, n=None):
    least_conf_images, least_conf_indices = least_confidence_images(embed_model, remainder, k=len(remainder) if 2*n > len(remainder) else 2*n)
    least_conf_embeddings = extract_embeddings(embed_model, least_conf_images)
    least_conf_embeddings = np.array([np.array(e) for e in least_conf_embeddings])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(least_conf_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_indices = get_most_diverse_samples(tsne_results, cluster_centers, n)
    diverse_images = data.Subset(least_conf_images, diverse_indices)
    return diverse_images, [least_conf_indices[i] for i in diverse_indices]

def LC_HC(model, remainder, n=5000):
    least_confident_2500, least_confident_indices = least_confidence_images(model, remainder, k=2500)
    most_confident_2500, most_confident_indices = high_confidence_images(model, remainder, k=2500)
    combined_dataset = data.ConcatDataset([least_confident_2500, most_confident_2500])
    combined_indices = least_confident_indices + most_confident_indices
    return combined_dataset, combined_indices

def LC_HC_diverse(embed_model, remainder, n, low_conf_ratio=0.5, high_conf_ratio=0.5):
    assert low_conf_ratio + high_conf_ratio == 1.0, "The sum of low_conf_ratio and high_conf_ratio must be 1.0"

    n_low = int(n * low_conf_ratio)
    n_high = int(n * high_conf_ratio)

    least_conf_images, least_conf_indices = least_confidence_images(embed_model, remainder, k=len(remainder) if 2*n_low > len(remainder) else 2*n_low)
    least_conf_embeddings = extract_embeddings(embed_model, least_conf_images)
    least_conf_embeddings = np.array([np.array(e) for e in least_conf_embeddings])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(least_conf_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_low_indices = get_most_diverse_samples(tsne_results, cluster_centers, n_low)
    diverse_least_conf_images = data.Subset(least_conf_images, diverse_low_indices)

    high_conf_images, high_conf_indices = high_confidence_images(embed_model, remainder, k=len(remainder) if 2*n_high > len(remainder) else 2*n_high)
    high_conf_embeddings = extract_embeddings(embed_model, high_conf_images)
    high_conf_embeddings = np.array([np.array(e) for e in high_conf_embeddings])
    tsne_results = tsne.fit_transform(high_conf_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_high_indices = get_most_diverse_samples(tsne_results, cluster_centers, n_high)
    diverse_high_conf_images = data.Subset(high_conf_images, diverse_high_indices)

    combined_dataset = data.ConcatDataset([diverse_least_conf_images, diverse_high_conf_images])
    combined_indices = [least_conf_indices[i] for i in diverse_low_indices] + [high_conf_indices[i] for i in diverse_high_indices]

    return combined_dataset, combined_indices


def train_until_empty(model, initial_train_set, remainder_set, test_set, max_iterations=15, batch_size=64, learning_rate=0.01, method=1):
    exp_acc = []
    
    for iteration in range(max_iterations):
        print(f"Starting Iteration {iteration}")
        print(f"Remaindersize:{len(remainder_set)}")
        if len(remainder_set) == 0:
            print("Dataset is empty. Stopping training.")
            break
            
        if method == 1:
            train_data,used_indices = LC_HC(model, remainder_set, n=5000)
        elif method == 2:
            train_data,used_indices = LC_HC_diverse(model, remainder_set, n=5000)
        elif method == 3:
            train_data,used_indices = HC_diverse(model, remainder_set, n=5000)
        elif method == 4:
            train_data,used_indices = LC_diverse(model, remainder_set, n=5000)
        else:
            print("Invalid Method")
            return exp_acc
    
        initial_train_set = data.ConcatDataset([initial_train_set, train_data])
        remainder_set = data.Subset(remainder_set, list(range(len(train_data), len(remainder_set))))
            
        print(f"\nTraining iteration {iteration + 1}")
        print(f"Train Set Size: {len(initial_train_set)}, Remainder Size: {len(remainder_set)}")
        train_loader = data.DataLoader(initial_train_set, batch_size=batch_size, shuffle=True)
        train_model(model, train_loader, epochs=50, learning_rate=learning_rate)

        test_loader = data.DataLoader(test_set, batch_size=batch_size)
        accuracy = test_model(model, test_loader)
        exp_acc.append(accuracy)

        print(f"Iteration {iteration + 1}: Test Accuracy - {accuracy}")

    return exp_acc

def run_all_methods(model, initial_train_set, remainder, test_set):
    methods = [1, 2, 3, 4]
    results = {}

    # Save initial model state
    initial_model_state = copy.deepcopy(model.state_dict())

    for method in methods:
        print(f"\nStarting training with method {method}")
        
        # Reset model to initial state
        model.load_state_dict(initial_model_state)
        
        # Create deep copies of datasets
        initial_train_set_copy = copy.deepcopy(initial_train_set)
        remainder_copy = copy.deepcopy(remainder)
        
        # Initial training
        train_loader = data.DataLoader(initial_train_set_copy, batch_size=64, shuffle=True)
        train_model(model, train_loader, epochs=1, learning_rate=0.01)
        
        # Initial testing
        test_loader = data.DataLoader(test_set, batch_size=64)
        initial_accuracy = test_model(model, test_loader)
        print(f"Initial accuracy for method {method}: {initial_accuracy}")
        
        # Run the active learning iterations
        exp_acc = train_until_empty(model, initial_train_set_copy, remainder_copy, test_set, 
                                  max_iterations=15, batch_size=64, learning_rate=0.01, method=method)
        results[f"method_{method}"] = exp_acc

    return results

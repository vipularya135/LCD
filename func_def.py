import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data
import numpy as np
import copy 
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import math
import os
from PIL import Image
from torch.utils.data import Dataset

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        self.images_dir = os.path.join(val_dir, 'images')
        self.annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        self.img_to_class = {}
        
        with open(self.annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name, class_name = parts[0], parts[1]
                    self.img_to_class[img_name] = class_name
        
        self.images = []
        for img_name in os.listdir(self.images_dir):
            if img_name in self.img_to_class:
                class_name = self.img_to_class[img_name]
                img_path = os.path.join(self.images_dir, img_name)
                self.images.append((img_path, class_name))
        
        self.classes = sorted(set(self.img_to_class.values()))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, class_name = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[class_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def prepare_data(dataset_name):
    print('==> Preparing data..')
    
    if dataset_name == 'tinyimagenet':
        data_dir = './data/tiny-imagenet-200'
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        train_set = datasets.ImageFolder(train_dir, transform=transform_train)
        test_set = TinyImageNetValDataset(val_dir, transform=transform_test)
        
        print(f"Tiny ImageNet Training Set Size: {len(train_set)}")
        print(f"Tiny ImageNet Validation Set Size: {len(test_set)}")

    else:
        raise ValueError("Invalid dataset name. Choose 'tinyimagenet'.")

    initial_size = int(0.04 * len(train_set))
    remainder_size = len(train_set) - initial_size
    initial_train_set, remainder = data.random_split(train_set, [initial_size, remainder_size])

    print(f"Size of initial_train_set: {len(initial_train_set)}")
    print(f"Size of remainder: {len(remainder)}")

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

def least_confidence_images(model, test_dataset, k=None):
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
    k = min(k, len(confidences)) if k is not None else len(confidences)
    _, indices = torch.topk(confidences, k, largest=False)
    return data.Subset(test_dataset, indices), indices.tolist()

def high_confidence_images(model, test_dataset, k=None):
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
    k = min(k, len(confidences)) if k is not None else len(confidences)
    _, indices = torch.topk(confidences, k, largest=True)
    return data.Subset(test_dataset, indices), indices.tolist()
    
def HC_diverse(embed_model, remainder, n=None):
    high_conf_images, high_conf_indices = high_confidence_images(embed_model, remainder, k=min(2*n, len(remainder)) if n else len(remainder))
    high_conf_embeddings = extract_embeddings(embed_model, high_conf_images)
    high_conf_embeddings = np.array([np.array(e) for e in high_conf_embeddings])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(high_conf_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_indices = get_most_diverse_samples(tsne_results, cluster_centers, n)
    diverse_images = data.Subset(high_conf_images, diverse_indices)
    return diverse_images, [high_conf_indices[i] for i in diverse_indices]

def LC_diverse(embed_model, remainder, n=None):
    least_conf_images, least_conf_indices = least_confidence_images(embed_model, remainder, k=min(2*n, len(remainder)) if n else len(remainder))
    least_conf_embeddings = extract_embeddings(embed_model, least_conf_images)
    least_conf_embeddings = np.array([np.array(e) for e in least_conf_embeddings])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(least_conf_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_indices = get_most_diverse_samples(tsne_results, cluster_centers, n)
    diverse_images = data.Subset(least_conf_images, diverse_indices)
    return diverse_images, [least_conf_indices[i] for i in diverse_indices]

def LC_HC(model, remainder, n=None):
    n_half = n // 2
    least_confident, least_confident_indices = least_confidence_images(model, remainder, k=n_half)
    most_confident, most_confident_indices = high_confidence_images(model, remainder, k=n_half)
    combined_dataset = data.ConcatDataset([least_confident, most_confident])
    combined_indices = least_confident_indices + most_confident_indices
    return combined_dataset, combined_indices

def LC_HC_diverse(embed_model, remainder, n, low_conf_ratio=0.5, high_conf_ratio=0.5):
    n_low = int(n * low_conf_ratio)
    n_high = int(n * high_conf_ratio)

    least_conf_images, least_conf_indices = least_confidence_images(embed_model, remainder, k=min(2*n_low, len(remainder)))
    least_conf_embeddings = extract_embeddings(embed_model, least_conf_images)
    least_conf_embeddings = np.array([np.array(e) for e in least_conf_embeddings])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(least_conf_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_low_indices = get_most_diverse_samples(tsne_results, cluster_centers, n_low)
    diverse_least_conf_images = data.Subset(least_conf_images, diverse_low_indices)

    high_conf_images, high_conf_indices = high_confidence_images(embed_model, remainder, k=min(2*n_high, len(remainder)))
    high_conf_embeddings = extract_embeddings(embed_model, high_conf_images)
    high_conf_embeddings = np.array([np.array(e) for e in high_conf_embeddings])
    tsne_results = tsne.fit_transform(high_conf_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_high_indices = get_most_diverse_samples(tsne_results, cluster_centers, n_high)
    diverse_high_conf_images = data.Subset(high_conf_images, diverse_high_indices)

    combined_dataset = data.ConcatDataset([diverse_least_conf_images, diverse_high_conf_images])
    combined_indices = [least_conf_indices[i] for i in diverse_low_indices] + [high_conf_indices[i] for i in diverse_high_indices]

    return combined_dataset, combined_indices

def train_until_empty(model, initial_train_set, remainder_set, test_set,
                      epochs=50, max_iterations=20, batch_size=32,
                      learning_rate=0.01, method=1):
    exp_acc = []
    original_dataset = remainder_set.dataset
    total_data_size = len(original_dataset)
    train_indices = set(initial_train_set.indices)
    remainder_indices = set(remainder_set.indices) - train_indices
    available_mask = np.ones(len(original_dataset), dtype=bool)
    available_mask[list(train_indices)] = False
    fixed_sample_size = int(0.05 * total_data_size)

    for iteration in range(max_iterations):
        current_remainder_indices = np.where(available_mask)[0]
        if len(current_remainder_indices) == 0:
            print("Dataset empty. Stopping.")
            break

        current_remainder = data.Subset(original_dataset, current_remainder_indices)
        print(f"\nStarting Iteration {iteration + 1}")
        print(f"Remainder Size: {len(current_remainder)}")

        if len(current_remainder) <= fixed_sample_size:
            relative_indices = list(range(len(current_remainder)))
            train_data = data.Subset(current_remainder.dataset,
                                     [current_remainder_indices[i] for i in relative_indices])
        else:
            if method == 1:
                train_data, relative_indices = LC_HC(model, current_remainder, n=fixed_sample_size)
            elif method == 2:
                train_data, relative_indices = LC_HC_diverse(model, current_remainder, n=fixed_sample_size)
            elif method == 3:
                train_data, relative_indices = HC_diverse(model, current_remainder, n=fixed_sample_size)
            elif method == 4:
                train_data, relative_indices = LC_diverse(model, current_remainder, n=fixed_sample_size)
            else:
                print("Invalid method.")
                return exp_acc

        absolute_indices = [current_remainder_indices[i] for i in relative_indices]
        available_mask[absolute_indices] = False

        samples_to_add = data.Subset(original_dataset, absolute_indices)
        all_train_indices = list(train_indices) + absolute_indices
        train_indices.update(absolute_indices)
        initial_train_set = data.Subset(original_dataset, all_train_indices)

        print(f"Train Size: {len(initial_train_set)}, Remainder Size: {len(current_remainder) - len(absolute_indices)}")

        train_loader = data.DataLoader(initial_train_set, batch_size=batch_size, shuffle=True)
        train_model(model, train_loader, epochs=epochs, learning_rate=learning_rate)

        test_loader = data.DataLoader(test_set, batch_size=batch_size)
        accuracy = test_model(model, test_loader)
        exp_acc.append(accuracy)
        print(f"Iteration {iteration + 1} Accuracy: {accuracy}")

    return exp_acc

def run_all_methods(model, initial_train_set, remainder, test_set):
    methods = [1, 2, 3, 4]
    results = {}
    initial_model_state = copy.deepcopy(model.state_dict())

    for method in methods:
        print(f"\nStarting training with method {method}")
        model.load_state_dict(initial_model_state)
        initial_train_set_copy = copy.deepcopy(initial_train_set)
        remainder_copy = copy.deepcopy(remainder)
        
        train_loader = data.DataLoader(initial_train_set_copy, batch_size=32, shuffle=True)
        print("\nStarting initial training with method")
        train_model(model, train_loader, epochs=50, learning_rate=0.01)
        
        test_loader = data.DataLoader(test_set, batch_size=64)
        initial_accuracy = test_model(model, test_loader)
        print(f"Initial accuracy for method {method}: {initial_accuracy}")
        
        exp_acc = train_until_empty(model, initial_train_set_copy, remainder_copy, test_set, 
                                  max_iterations=20, batch_size=32, learning_rate=0.01, method=method)
        results[f"method_{method}"] = exp_acc

    return results
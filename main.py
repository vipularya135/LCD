from models import *
from func_def import *
import torch
import os
import copy 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42)
    
    initial_train_set, remainder, test_set = prepare_data("tinyimagenet") 

    all_models = {
        'resnet18': ResNet(BasicBlock, [2, 2, 2, 2], num_classes=200).to(device),
        'resnet50': ResNet(Bottleneck2, [3, 4, 6, 3], num_classes=200).to(device),
        'resnet56': ResNet(BasicBlock, [9, 9, 9, 9], num_classes=200).to(device),
        'mobilenet': MobileNet(num_classes=200).to(device),
        'densenet121': DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_classes=200).to(device),
        'vgg16': VGG16(num_classes=200).to(device)
    }

    with open('results.txt', 'w') as f:
        for name, model in all_models.items():
            print(f"\nRunning model: {name}")
            model_results = run_all_methods(model, initial_train_set, remainder, test_set)
            
            f.write(f'Model: {name}\n')
            f.write('=' * (7 + len(name)) + '\n\n')
            for method, accuracies in model_results.items():
                f.write(f'{method}:\n')
                f.write('-' * len(method) + '\n')
                for i, acc in enumerate(accuracies):
                    f.write(f'Iteration {i+1}: {acc:.4f}\n')
                f.write('\n')
            f.write('\n\n')

if __name__ == "__main__":
    main()
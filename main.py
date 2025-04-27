from models import *
from swin import *
from vit_tiny import *
from func_def import prepare_data, run_all_methods

import torch

def main():
    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42)
    
    # Prepare data
    initial_train_set, remainder, test_set = prepare_data("cifar10") 

    # All models dictionary
    all_models = {
        # 'resnet18': ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device),
        # 'resnet50': ResNet(Bottleneck2, [3, 4, 6, 3], num_classes=10).to(device),
        # 'resnet56': ResNet(BasicBlock, [9, 9, 9, 9], num_classes=10).to(device),
        # 'mobilenet': MobileNet(num_classes=10).to(device),
        # 'densenet121': DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_classes=10).to(device),
        'vgg16': VGG16(num_classes=10).to(device)
        # 'swin': SwinTransformer(
        #     img_size=32, num_classes=10, window_size=4, patch_size=2,
        #     embed_dim=96, depths=[2, 6, 4], num_heads=[3, 6, 12],
        #     mlp_ratio=2, qkv_bias=True, drop_path_rate=0.1
        # ).to(device)
        # 'vit' :  ViT(
        #     image_size = 32,
        #     patch_size = 4,
        #     num_classes = 10,
        #     dim = 256,
        #     depth = 6,
        #     heads = 8,
        #     mlp_dim = 512,
        #     dim_head = 32,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # ).to(device)
    }

    # Open the output file
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

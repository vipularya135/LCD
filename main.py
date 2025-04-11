from models import DenseNet, Bottleneck
from func_def import prepare_data, train_until_empty
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

initial_train_set, remainder, test_set = prepare_data("cifar10")

model = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_classes=10).to(device)

results = train_until_empty(
        model=model,
        initial_train_set=initial_train_set,
        remainder_set=remainder,
        test_set=test_set,
        max_iterations=2,
        batch_size=64,
        learning_rate=0.01,
        method=1
    )

    # Output results
for i, acc in enumerate(results):
        print(f"Iteration {i+1}: {acc:.4f}")



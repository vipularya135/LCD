import torch
from your_model_file import RetinaNet  # replace with the actual filename if saved

# 1. Instantiate the model
model = RetinaNet(num_classes=80)  # e.g., 80 for COCO dataset

# 2. Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 3. Create a dummy input image tensor (e.g., batch size 1, 3x512x512 image)
dummy_input = torch.randn(1, 3, 512, 512).to(device)

# 4. Forward pass through RetinaNet
loc_preds, cls_preds = model(dummy_input)

# 5. Print output shapes
print("Localization predictions shape:", loc_preds.shape)  # [batch_size, num_anchors, 4]
print("Classification predictions shape:", cls_preds.shape)  # [batch_size, num_anchors, num_classes]

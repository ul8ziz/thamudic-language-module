import torch

# Path to the checkpoint file
checkpoint_path = '../models/best_model.keras'

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Print the keys in the checkpoint
print('Checkpoint Keys:', checkpoint.keys())

# Print the model state dict keys if available
if 'model_state_dict' in checkpoint:
    print('Model State Dict Keys:', checkpoint['model_state_dict'].keys())
else:
    print('No model_state_dict found in checkpoint')

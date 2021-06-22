import torch

model_save_path = 'model/binary_model_5d_2.pth'

layer_sizes = {
    'layer_0': [X_train.shape[1], 333, True, nn.PReLU, 0],
    'layer_1': [333, 166, True, nn.PReLU, 0],
    'layer_2': [166, 33, False, nn.GELU, 0],
    'layer_5': [33, 5, True, None, 0]
} 

model = NN_split(layer_sizes).to(device)
model.load_state_dict(torch.load(model_save_path))

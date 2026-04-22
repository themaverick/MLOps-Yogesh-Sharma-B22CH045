
import torch

device = torch.device("cpu")

model = YourModelClass()  # <-- define architecture exactly as training
model.load_state_dict(torch.load("model_final.pth", map_location=device))
model.to(device)
model.eval()

import torch

device = torch.device("cpu")

checkpoint = torch.load("checkpoint_latest.pth", map_location=device)

model = YourModelClass()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# (optional) restore optimizer if you want to resume training
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"] + 1

train_losses = checkpoint["train_losses"]
val_ious = checkpoint["val_ious"]
val_dices = checkpoint["val_dices"]
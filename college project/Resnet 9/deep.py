# %%
# Upload model weights file (resnet9-deepfake-detector.pth) if present
from google.colab import files
uploaded = files.upload()

# %%
!pip install opendatasets --upgrade --quiet

# %%
import os
import zipfile
import gc
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import opendatasets as od
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

matplotlib.rcParams['figure.facecolor'] = '#ffffff'

# %%
od.download(
    "https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images",
    data_dir="data"
)

# %%
data_dir = './data/deepfake-and-real-images/Dataset'

# Look inside the dataset directory
print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/Train")
print(classes)

# %%
# dataset = datasets.ImageFolder(root=data_dir + "/Train", transform=transforms.ToTensor())
# dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=2)

# mean = torch.zeros(3)
# std = torch.zeros(3)
# nb_samples = 0.

# for data in dataloader:
#   images, _ = data
#   batch_samples = images.size(0)
#   images = images.view(batch_samples, images.size(1), -1)
#   mean += images.mean(2).sum(0)
#   std += images.std(2).sum(0)
#   nb_samples += batch_samples

# mean /= nb_samples
# std /= nb_samples

# print("Mean:", mean)
# print("Standard Deviation:", std)

# stats = (mean.tolist(), std.tolist())

# %%
stats = ((0.4668, 0.3816, 0.3414), (0.2410, 0.2161, 0.2081))

# %%
# Data transforms (normalization & data augmentation)
train_tfms = tt.Compose([
    tt.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
    tt.Normalize(*stats, inplace=True)
])

valid_tfms = tt.Compose([
    tt.Resize((64, 64)),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

test_tfms = tt.Compose([
    tt.Resize((64, 64)),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

# %%
# PyTorch datasets
train_ds = ImageFolder(data_dir+'/Train', train_tfms)
valid_ds = ImageFolder(data_dir+'/Validation', valid_tfms)
test_ds = ImageFolder(data_dir+'/Test', test_tfms)
# new_ds = ImageFolder(data_dir+'/New', test_tfms)

# %%
print(train_ds.classes)

# %%
batch_size = 400

# %%
# PyTorch data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                      num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=2, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size*2, num_workers=2, pin_memory=True)

# %%
def denormalize(images, means, stds):
  means = torch.tensor(means).reshape(1, 3, 1, 1)
  stds = torch.tensor(stds).reshape(1, 3, 1, 1)
  return images * stds + means

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        denorm_images = denormalize(images, *stats)
        ax.imshow(make_grid(denorm_images[:64], nrow = 8).permute(1, 2, 0).clamp(0,1))
        break

# %%
show_batch(train_dl)

# %%
def get_default_device():
  """Pick GPU if available, else CPU"""
  if torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")

def to_device(data, device):
  """Move tensors to choosen device"""
  if isinstance(data, (list, tuple)):
    return [to_device(x, device ) for x in data]
  return data.to(device, non_blocking=True)

class DeviceDataLoader():
  """Wrap a dataloader to move data to a device"""
  def __init__(self, dl, device):
    self.dl = dl
    self.device = device

  def __iter__(self):
    """Yield a batch of data after moving it to device"""
    for b in self.dl:
      yield to_device(b, self.device)

  def __len__(self):
    """Number of batches"""
    return len(self.dl)

# %%
device = get_default_device()
device

# %%
trian_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
test_dl = DeviceDataLoader(test_dl, device)

# %%
class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x

# %%
simple_resnet = SimpleResidualBlock()
simple_resnet = to_device(simple_resnet, device)

for images, labels in train_dl:
    print(images.shape)
    images = to_device(images, device)
    out = simple_resnet(images)
    print(out.shape)
    break

del simple_resnet, images, labels
torch.cuda.empty_cache()

# %%
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

# %%
def conv_block(in_channels, out_channels, pool=False):
  layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]
  if pool: layers.append(nn.MaxPool2d(2))
  return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
  def __init__(self, in_channels, num_classes):
    super().__init__()
    # 3 * 64 * 64
    self.conv1 = conv_block(in_channels, 64)             # 64 * 64 * 64
    self.conv2 = conv_block(64, 128, pool=True)          # 128 * 32 * 32
    self.res1 = nn.Sequential(conv_block(128, 128),
                              conv_block(128, 128))       # 128 * 32 * 32

    self.conv3 = conv_block(128, 256, pool=True)         # 256 * 16 * 16
    self.conv4 = conv_block(256, 512, pool=True)         # 512 * 8 * 8
    self.res2 = nn.Sequential(conv_block(512, 512),
                              conv_block(512, 512))       # 512 * 8 * 8

    self.classifier = nn.Sequential(nn.MaxPool2d(4),      # 512 * 2 * 2
                                    nn.Flatten(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512 * 2 * 2, num_classes))  # 2

  def forward(self, xb):
    out = self.conv1(xb)
    out = self.conv2(out)
    out = self.res1(out) + out
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.res2(out) + out
    out = self.classifier(out)
    return out

# %%
model = to_device(ResNet9(3, 2), device)

# %%
model.load_state_dict(torch.load('resnet9-deepfake-detector.pth', map_location=device))

# %%
model.eval()

# %% [markdown]
# Training the model

# %%
@torch.no_grad()
def evaluate(model, val_loader):
  model.eval()
  outputs = []
  for batch in val_loader:
    batch = to_device(batch, device)
    out = model.validation_step(batch)
    outputs.append(out)
    del batch
    torch.cuda.empty_cache()
    gc.collect()

  return model.validation_epoch_end(outputs)

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
  torch.cuda.empty_cache()
  history = []

  # Set up custom optimizer with weight decay
  optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
  # Set up one-cycle learning rate scheduler
  sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

  for epoch in range(epochs):
    # Training phase
    model.train()
    train_losses = []
    lrs = []
    for batch in train_loader:
      batch = to_device(batch, device)
      loss = model.training_step(batch)
      train_losses.append(loss)
      loss.backward()

      # Gradient clipping
      if grad_clip:
        nn.utils.clip_grad_value_(model.parameters(), grad_clip)

      optimizer.step()
      optimizer.zero_grad()

      # Record and update learing rate
      lrs.append(get_lr(optimizer))
      sched.step()

      del batch
      torch.cuda.empty_cache()
      gc.collect()

    # Validation phase
    result = evaluate(model, val_loader)
    result['train_loss'] = torch.stack(train_losses).mean().item()
    result['lrs'] = lrs
    model.epoch_end(epoch, result)
    history.append(result)
  return history

# %%
history = [evaluate(model, valid_dl)]
history

# %%
# # Training the model with these parameters

# epochs = 8
# max_lr = 0.01
# grad_clip = 0.1
# weight_decay = 1e-4
# opt_func = torch.optim.Adam

# %%time
# history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, grad_clip=grad_clip, weight_decay=weight_decay, opt_func=opt_func)

# %%
# train_time = '47:46'

# Epoch [0], train_loss: 0.4564, val_loss: 0.9060, val_acc: 0.7266
# Epoch [1], train_loss: 0.2182, val_loss: 0.5837, val_acc: 0.7448
# Epoch [2], train_loss: 0.1351, val_loss: 1.0577, val_acc: 0.6569
# Epoch [3], train_loss: 0.1142, val_loss: 0.2754, val_acc: 0.8841
# Epoch [4], train_loss: 0.0978, val_loss: 0.9434, val_acc: 0.7107
# Epoch [5], train_loss: 0.0901, val_loss: 0.1217, val_acc: 0.9518
# Epoch [6], train_loss: 0.0621, val_loss: 0.0861, val_acc: 0.9664
# Epoch [7], train_loss: 0.0466, val_loss: 0.0890, val_acc: 0.9648
# CPU times: user 36min 8s, sys: 1min 28s, total: 37min 37s
# Wall time: 47min 46s

# %%
# # Save the model
# torch.save(model.state_dict(), 'resnet9-deepfake-detector.pth')

# %%
def plot_accuracies(history):
  accuracies = [x['val_acc'] for x in history]
  plt.plot(accuracies, '-x')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.title('Accuracy vs Number of epochs')

# %%
plot_accuracies(history)

# %%
def plot_losses(history):
  train_losses = [x.get('train_loss') for x in history]
  val_losses = [x['val_loss'] for x in history]
  plt.plot(train_losses, '-bx')
  plt.plot(val_losses, '-rx')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend(['Training', 'Validation'])
  plt.title('Loss vs Number of epochs')

# %%
plot_losses(history)

# %%
def plot_lrs(history):
  lrs = np.concatenate([x.get('lrs', []) for x in history])
  plt.plot(lrs)
  plt.xlabel('Batch Number')
  plt.ylabel('Learning rate')
  plt.title('Learing rate vs Batch number')

# %%
plot_lrs(history)

# %% [markdown]
# Testing the model

# %%
test_history = [evaluate(model, test_dl)]
test_history

# %% [markdown]
# Testing with individual images

# %%
def predict_image(img, model):
  # Convert to a batch of 1
  xb = to_device(img.unsqueeze(0), device)
  # Get prdictions from model
  yb = model(xb)
  # Pick index with highest probability
  _, preds = torch.max(yb, dim=1)
  # Retrieve the class label
  return train_ds.classes[preds[0].item()]

# %%
img, label = valid_ds[0]
plt.imshow(img.permute(1, 2, 0).clamp(0, 1))
print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))

# %%
img, label = valid_ds[1003]
plt.imshow(img.permute(1, 2, 0).clamp(0, 1))
print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))

# %%
img, label = valid_ds[19642]
plt.imshow(img.permute(1, 2, 0).clamp(0, 1))
print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))

# %% [markdown]
# Checking new images by uploading them to "New" folder in Dataset folder

# %%
import torch
from PIL import Image
import os

# 1️⃣  Map indices back to class names (adjust if your order differs)
idx_to_class = {0: 'Fake', 1: 'Real'}

# 2️⃣  Device‑aware prediction helper
def predict_image(model, image_path, transform, idx_to_class):
    """
    Returns 'Real' or 'Fake' for a single image.
    Works whether the model is on CPU or GPU.
    """
    # a) Get model’s current device
    device = next(model.parameters()).device           # cpu or cuda:0
    # b) Preprocess image and move it to that device
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # add batch dim & move
    # c) Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        # CrossEntropyLoss case (two‑class softmax)
        _, pred_idx = torch.max(logits, 1)
        label = idx_to_class[pred_idx.item()]
        # --- If you used sigmoid/BCELoss, swap for: ---
        # prob = torch.sigmoid(logits).item()
        # label = 'Fake' if prob > 0.5 else 'Real'
    return label

# 3️⃣  Loop through the images in New/
new_dir = os.path.join(data_dir, 'New')
image_files = [f for f in os.listdir(new_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]

for img_name in image_files:
    img_path = os.path.join(new_dir, img_name)
    pred = predict_image(model, img_path, test_tfms, idx_to_class)
    print(f"{img_name}: {pred}")




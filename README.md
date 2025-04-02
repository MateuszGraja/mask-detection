# Face Mask Detection with ResNet-Like CNN

---

## Project Description

This project trains a **deep convolutional neural network (ResNet-inspired)** to classify images of people based on mask usage. It uses the public *Face Mask Detection* dataset from Kaggle, featuring images with annotations in Pascal VOC XML format.

**Categories:**
- `with_mask`
- `without_mask`
- `mask_weared_incorrect`

## Key Features

- ✅ Custom PyTorch `Dataset` with XML label parsing
- ✅ Automatic download from KaggleHub
- ✅ Custom-built ResNet-like architecture with residual connections
- ✅ Train/test split with evaluation
- ✅ High-quality preprocessing using `torchvision.transforms`

---

## Dataset

**Kaggle Source**: [andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

After downloading with KaggleHub:
```
📂 /face-mask-detection
├── annotations/  ← .xml files (labels)
├── images/       ← .png files (input images)
```

---

## Model Architecture

The network follows a custom ResNet-like structure:
```python
class ResNet_Like(nn.Module):
    def __init__(self):
        self.block1 = ResBlock(3, 32, 32)
        self.block2 = ResBlock(32, 128, 128)
        self.block3 = ResBlock(128, 256, 256)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(256, 3)
```
Each `ResBlock` consists of multiple convolutional layers and a skip connection:
```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        self.identity_mapping = nn.Conv2d(in_channels, out_channels, 1)
        ...
        def forward(x):
            identity = self.identity_mapping(x)
            x = self.cnn1(x)
            x = self.cnn2(x)
            x = self.cnn3(x + identity)
            return pool(x)
```

---

## Training Configuration

```python
learning_rate = 3e-4
batch_size = 32
epochs = 10
```

Images are resized to 224x224 and normalized using ImageNet means:
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

Training loop:
```python
for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        ...
```

Evaluation:
```python
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        accuracy = correct / total
```

---

## Results

Sample training log:
```
Epoch [1/10], Loss: 0.8342
Epoch [2/10], Loss: 0.5421
...
92.75%
```
---

## Requirements

- Python 3.8+
- torch
- torchvision
- kagglehub
- Pillow

Install with:
```bash
pip install torch torchvision kagglehub Pillow
```

import kagglehub
import os
import torch
import torch.nn as nn
import torch.optim as optim
import xml.etree.ElementTree as ET
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
print("Path to dataset files:", path)

def read_xml_data(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels = []
    for member in root.findall('object'):
        label = member.find('name').text
        labels.append(label)
    return labels[0] if labels else "unknown"

class MaskData(Dataset):
    def __init__(self, annotation_dir, images_dir, transform=None):
        self.images_dir = images_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        annotation_file = os.path.join(self.annotation_dir, self.image_files[idx].replace('.png', '.xml'))
        label = read_xml_data(annotation_file)

        if self.transform:
            image = self.transform(image)

        label = self.label_to_int(label)
        return image, label

    def label_to_int(self, label):
        label_map = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}
        return label_map.get(label, -1)

# Hyperparameters
learning_rate = 3e-4
batch_size = 32
num_epochs = 10

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = MaskData(
    annotation_dir=path+'/annotations/',
    images_dir=path+'/images/',
    transform=transform,
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_load = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_load = DataLoader(test_set, batch_size=batch_size, shuffle=True)

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.ReLU()(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResBlock, self).__init__()
        self.cnn1 = CNNBlock(in_channels, mid_channels)
        self.cnn2 = CNNBlock(mid_channels, mid_channels)
        self.cnn3 = CNNBlock(mid_channels, out_channels)
        self.identity_mapping = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.identity_mapping(x)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x + identity)
        x = self.pooling(x)
        return x

class ResNet_Like(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet_Like, self).__init__()
        self.block1 = ResBlock(in_channels=3, mid_channels=32, out_channels=32)
        self.block2 = ResBlock(in_channels=32, mid_channels=128, out_channels=128)
        self.block3 = ResBlock(in_channels=128, mid_channels=256, out_channels=256)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet_Like(num_classes=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Trening
model.train()
for epoch in range(num_epochs):
    run_loss = 0.0
    for images, labels in train_load:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {run_loss / len(train_load):.4f}")

# Ewaluating
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_load:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"{100 * correct / total:.2f}%")

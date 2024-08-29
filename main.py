import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchinfo import summary
from dataset_create import LinePlotDataset
from model import CNNLSTMModel

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Augmentation
transform = A.Compose([
    A.Resize(100, 100),
    A.RandomRotate90(),
    A.Flip(),
    A.RandomBrightnessContrast(),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

dataset = LinePlotDataset(transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate the model and move it to the GPU
model = CNNLSTMModel().to(device)

print(summary(model,input_size=(1, 3, 100, 100)))
# Step 4: Training Setup
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Model Training
num_epochs = 20
for epoch in range(num_epochs):
    for images, coords in dataloader:
        
        # Move inputs and targets to the GPU
        images = images.to(device)
        coords = coords.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, coords)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 6: Inference with Post-processing
def preprocess_image(img):
    # Standardize to expected input format (BGR to RGB if using PyTorch)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure color channels are in RGB order
    img = cv2.resize(img, (100, 100))  # Resize to model's expected input size
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

def predict_coordinates(model, img):
    img = preprocess_image(img)
    
    # Convert image to tensor and move to GPU
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)  # Prepare for model (1, 3, 100, 100)
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Perform inference
    with torch.no_grad():
        pred = model(img)
    
    # Move the prediction back to the CPU and convert to numpy
    pred_coords = pred.cpu().squeeze(0).numpy()
    return pred_coords

# Testing with a sample image
sample_image, _ = dataset[0]  # Get the first sample image and its coordinates
predicted_coords = predict_coordinates(model, sample_image.numpy().transpose(1, 2, 0))

# Visualization
plt.imshow(sample_image.numpy().transpose(1, 2, 0))
print(predicted_coords)

plt.scatter(predicted_coords[:, 0], predicted_coords[:, 1], color='red')
plt.savefig('/home/ec2-user/aniket/Datapoint/predicted_coordinates.png', bbox_inches='tight')

plt.show()

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import numpy as np
import cv2
from dataset_create import LinePlotDataset
from model import ModifiedCNNModel 

# Function to train the model
def train_model(model, train_loader, eval_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)  # Transfer model to the specified device (GPU/CPU)
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for images, true_coords in train_loader:
            images, true_coords = images.to(device), true_coords.to(device)  # Transfer data to the device
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images.float())  # Convert images to float
            loss = criterion(outputs, true_coords)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average loss over the epoch
        avg_train_loss = running_loss / len(train_loader)
        
        # Evaluate the model
        eval_loss = evaluate_model(model, eval_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

# Function to evaluate the model
def evaluate_model(model, eval_loader, criterion, device='cpu'):
    model.eval()  # Set model to evaluation mode
    eval_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for images, true_coords in eval_loader:
            images, true_coords = images.to(device), true_coords.to(device)  # Transfer data to the device
            outputs = model(images.float())
            loss = criterion(outputs, true_coords)
            eval_loss += loss.item()
    
    avg_eval_loss = eval_loss / len(eval_loader)
    return avg_eval_loss

def draw_coordinates(image, coords, color, thickness=1):
    """
    Draws lines between coordinates on the image.
    :param image: The image to draw on.
    :param coords: The coordinates to draw, shape [num_points, 2].
    :param color: The color of the lines (B, G, R).
    :param thickness: The thickness of the lines.
    """
    points = np.array([list(map(int, p)) for p in coords])
    for i in range(len(points) - 1):
        cv2.line(image, tuple(points[i]), tuple(points[i + 1]), color, thickness)

def test_model(model, dataset, device='cpu', save_path='output.png'):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        image, true_coords = dataset[0]  # Test on the first example
        image_tensor = torch.tensor(image).unsqueeze(0).float().to(device)  # Add batch dimension, convert to float, and transfer to device
        output = model(image_tensor)
        predicted_coords = output.squeeze().cpu().numpy()  # Transfer to CPU and convert to numpy
        
        # Convert image back to original format for OpenCV (HWC)
        # image = np.transpose(image, (1, 2, 0))
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White background

        # Draw the true coordinates in green
        draw_coordinates(image, true_coords, color=(0, 255, 0))
        
        # Draw the predicted coordinates in red
        draw_coordinates(image, predicted_coords, color=(0, 0, 255))
        
        # Save the image with the drawn coordinates
        cv2.imwrite(save_path, image)
        
        print(f"Image saved to {save_path}")
        print("True Coordinates:", true_coords)
        print("Predicted Coordinates:", predicted_coords)

dataset = LinePlotDataset(num_samples=1000)
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

# Determine if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, and Optimizer setup
model = ModifiedCNNModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model with 10 epochs (example)
train_model(model, train_loader, eval_loader, criterion, optimizer, num_epochs=50, device=device)

# Test the model on a single example from the dataset
test_model(model, dataset, device=device, save_path='output.png')

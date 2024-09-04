from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

class LinePlotDataset(Dataset):
    def __init__(self, num_samples=1000, transform=None):
        self.images, self.coordinates = self.generate_synthetic_data(num_samples)
        self.transform = transform

    def generate_synthetic_data(self, num_samples):
        images = []
        coordinates = []
        for _ in range(num_samples):
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White background
            x = np.linspace(10, 90, 30)  # 30 points for the line
            y = np.random.uniform(10, 90, size=30)
            coordinates.append(np.vstack((x, y)).T)

            # Draw the line with consistent color and width
            points = np.array([list(map(int, p)) for p in zip(x, y)])
            for i in range(len(points) - 1):
                cv2.line(img, tuple(points[i]), tuple(points[i + 1]), (70, 130, 180), 2)  # Blue color, width 2

            images.append(img)
        return np.array(images), np.array(coordinates)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        coords = self.coordinates[idx].astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Convert from HWC to CHW format
        image = np.transpose(image, (2, 0, 1))

        return image, coords


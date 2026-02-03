import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define transformations: Data Augmentation + Normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # Random Horizontal Flip
    transforms.RandomRotation(10),            # Random rotation within 10 degrees
    transforms.Resize(224),                   # Resize images to 224x224
    transforms.ToTensor(),                    # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization (ImageNet statistics)
                         std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset with transformations
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Set the random seed for reproducibility
random_seed = 10
torch.manual_seed(random_seed)

# Split dataset into training (80%) and testing (20%) sets
train_size = int(0.8 * len(train_dataset))  # 80% for training
test_size = len(train_dataset) - train_size  # 20% for testing
train_data, test_data = random_split(train_dataset, [train_size, test_size],
                                     generator=torch.Generator().manual_seed(random_seed))

# DataLoader for batching
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

# Load a pretrained AlexNet model and modify for CIFAR-10
model = models.alexnet(pretrained=True)
# model = models.vgg16(pretrained=True)
# model = models.resnet18(pretrained=True)
model.classifier[6] = nn.Linear(4096, 10)  # Adjust for 10 classes

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer (Adam) and loss function (Cross-Entropy Loss)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Early stopping parameters
patience = 5
best_train_loss = np.inf
epochs_without_improvement = 0

num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    total = 0
    correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Zero gradients
        
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)
        
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize
        
        running_loss += loss.item()

        # Calculate accuracy on the training batch
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate average training loss and accuracy for the epoch
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
    
    # Early stopping logic based on training loss
    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'alexnet.pth')
        print("Model improved. Saving model.")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break
    
    # Step the scheduler
    scheduler.step()

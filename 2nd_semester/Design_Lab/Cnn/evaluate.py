import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Define the same transformations as in train.py
transform = transforms.Compose([
    transforms.Resize(224),  # Resize images to 224x224 (no augmentation needed for evaluation)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset with transformations
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Set the random seed for reproducibility
random_seed = 10

# Recreate the same test split as in train.py
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_data = random_split(dataset, [train_size, test_size],
                            generator=torch.Generator().manual_seed(random_seed))

# DataLoader for the test set
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Load the pretrained AlexNet model and modify for CIFAR-10
model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096, 10)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the best model weights saved during training
# model.load_state_dict(torch.load('alexnet.pth'))
model.load_state_dict(torch.load('alexnet.pth', map_location=torch.device('cpu')))
model.eval()

# Function for model evaluation
def evaluate_model(model, test_loader, device):
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix
    conf_matrix_df = pd.DataFrame(conf_matrix, index=[f'Class {i}' for i in range(10)], columns=[f'Pred {i}' for i in range(10)])
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

    # Generate PDF report
    def generate_report():
        pdf_file = "alexnet_report.pdf"
        c = canvas.Canvas(pdf_file, pagesize=letter)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(30, 750, "Model Evaluation Report")
        c.setFont("Helvetica", 12)
        c.drawString(30, 720, f"Accuracy: {accuracy:.4f}")
        c.drawString(30, 705, f"Precision: {precision:.4f}")
        c.drawString(30, 690, f"Recall: {recall:.4f}")
        c.drawString(30, 675, f"F1-Score: {f1:.4f}")

        # Save the confusion matrix as an image file
        conf_matrix_fig = plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close(conf_matrix_fig)

        # Add confusion matrix image to the PDF
        c.drawImage('confusion_matrix.png', 30, 400, width=500, height=300)
        c.save()
        print(f"Report saved as {pdf_file}")

    generate_report()

# Run evaluation
evaluate_model(model, test_loader, device)

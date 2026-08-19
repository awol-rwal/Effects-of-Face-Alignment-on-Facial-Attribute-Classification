import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import re

# Alter the following values
MODEL_NAME = "resnet" # Choose from "resnet", "vgg" or "efficientnet"
TRAIN_PATH = "aligned_train"
VALIDATE_PATH = "aligned_validate"
TEST_PATH = "aligned_test"

# Constants
NUM_ATTRIBUTES = 40
BATCH_SIZE = 64

# Globals
criterion = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_labels(path):
    label_fobj = open(f"{path}/labels.txt")
    label_names = label_fobj.readline()
    labels = []
    img_names = []
    while True:
        line = label_fobj.readline()
        if not line:
            break
        raw_labels = re.split(r'\s+', line.strip())
        img_names.append(raw_labels.pop(0))
        clean_labels = [int(s) for s in raw_labels]

        labels.append(clean_labels)

    return img_names, labels



class CustomDataset(Dataset):
    def __init__(self, labels, img_dir, img_names, transform=None):
        self.img_dir = img_dir
        self.img_names = img_names
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


def vgg():
    # Load model
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Final Layer Replacement
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(in_features, NUM_ATTRIBUTES),
        nn.Tanh()
    )


    # Freeze all weights
    for p in model.parameters():
        p.requires_grad = False


    # Unfreeze the final layer weights
    # classifier[6] is the final Sequential we set
    for p in model.classifier[6].parameters():
        p.requires_grad = True

    return model


def resnet():
    # Load Model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Final Layer Replacement
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, NUM_ATTRIBUTES),
        nn.Tanh()
    )

    # Freeze all weights
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze the final layer weights
    for p in model.fc.parameters():
        p.requires_grad = True

    return model


def efficientnet():
    # Load Model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Final Layer Replacement
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, NUM_ATTRIBUTES),
        nn.Tanh()
    )

    # Freeze all weights
    for p in model.parameters():
        p.requires_grad = False


    # Unfreeze the final layer weights
    # classifier[1] is the final Sequential we set
    for p in model.classifier[1].parameters():
        p.requires_grad = True

    return model


def train(model):
    train_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    training_accuracy = []
    training_precision = []
    training_recall = []
    training_f1 = []
    training_loss = []
    
    # Training
    num_epochs = 5
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

        # Validation
        average_accuracy, average_loss, precision, recall, f1_score = evaluate(model, validate_dataset)
        training_accuracy.append(average_accuracy)
        training_precision.append(precision)
        training_recall.append(recall)
        training_f1.append(f1_score)
        training_loss.append(average_loss)

    print("accuracy:")
    print(training_accuracy)
    print("precision:")
    print(training_precision)
    print("recall:")
    print(training_recall)
    print("f1:")
    print(training_f1)
    print("loss:")
    print(training_loss)


def evaluate(model, dataset):
    """
    Evaluates the model on the given dataset, calculating loss, accuracy,
    precision, recall, and F1-score.
    """
    # Create a DataLoader for the test set
    dataset_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    total_loss = 0
    
    # --- New Trackers ---
    # We'll calculate metrics for the "positive" class (1.0)
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    tn = 0  # True Negatives
    # --- End New Trackers ---

    THRESHOLD = 0.0 

    with torch.no_grad():
        for inputs, labels in dataset_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # --- Loss Calculation (MSE Loss) ---
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # --- Metrics Calculation ---
            predicted = torch.where(outputs > THRESHOLD, 
                                    torch.tensor(1.0, device=device), 
                                    torch.tensor(-1.0, device=device))

            # --- Update TP, FP, FN, TN ---
            # (predicted == 1.0) & (labels == 1.0) means True Positive
            tp += ((predicted == 1.0) & (labels == 1.0)).sum().item()
            
            # (predicted == 1.0) & (labels == -1.0) means False Positive
            fp += ((predicted == 1.0) & (labels == -1.0)).sum().item()
            
            # (predicted == -1.0) & (labels == 1.0) means False Negative
            fn += ((predicted == -1.0) & (labels == 1.0)).sum().item()
            
            # (predicted == -1.0) & (labels == -1.0) means True Negative
            tn += ((predicted == -1.0) & (labels == -1.0)).sum().item()
            # --- End Update ---

    # --- Final Metric Calculations ---
    average_loss = total_loss / len(dataset_loader)

    # Calculate overall accuracy
    # (This is the same as your original 'average_accuracy')
    total_attributes = tp + tn + fp + fn
    average_accuracy = (tp + tn) / total_attributes

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-7 

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp + epsilon)
    
    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn + epsilon)
    
    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    print(f"Average Test Loss: {average_loss:.4f}")
    print(f"Test Multi-Label Attribute Accuracy: {average_accuracy:.4f}")
    print("---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print("---")
    print(f"(TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn})")

    # Return the new metrics
    return average_accuracy, average_loss, precision, recall, f1_score


# --- Main Execution ---
if __name__ == "__main__":
    if MODEL_NAME == "resnet":
        transform = models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    elif MODEL_NAME == "vgg":
        transform = models.VGG16_Weights.IMAGENET1K_V1.transforms()
    elif MODEL_NAME == "efficientnet":
        transform = models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    # Make sure to update all necessary directory paths below
    # Load labels
    test_labels = []
    validate_labels = []
    train_labels = []
    train_img_names, train_labels = load_labels(TRAIN_PATH)
    validate_img_names, validate_labels = load_labels(VALIDATE_PATH)
    test_img_names, test_labels = load_labels(TEST_PATH)

    # Load Datasets
    training_dataset = CustomDataset(train_labels, TRAIN_PATH, train_img_names, transform=transform)
    validate_dataset = CustomDataset(validate_labels, VALIDATE_PATH, validate_img_names, transform=transform)
    testing_dataset = CustomDataset(test_labels, TEST_PATH, test_img_names, transform=transform)
    
    # Initialise Model
    if MODEL_NAME == "resnet":
        model = resnet()
        print("New ResNet final layer:\n", model.fc)
    elif MODEL_NAME == "vgg":
        model = vgg()
        print("New VGG final layer:\n", model.classifier[6])
    elif MODEL_NAME == "efficientnet":
        model = efficientnet()
        print("New EfficientNet final layer:\n", model.classifier[1])
    
    model.to(device)

    print("TRAINING")
    train(model)
    print("TESTING")
    evaluate(model, testing_dataset)

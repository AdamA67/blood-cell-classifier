import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import build_model
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# ── Config ──────────────────────────────────────────────
DATA_DIR   = "data/bloodcells_dataset"
BATCH_SIZE = 32
EPOCHS     = 5
LR         = 0.001
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

def load_data(train_tf, val_tf):
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    

    n = len(full_dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test]
    )
    
    # Apply val transforms to val and test
    val_set.dataset.transform  = val_tf
    test_set.dataset.transform = val_tf
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE)
    
    return train_loader, val_loader, test_loader, full_dataset.classes

def train_epoch(model, loader, criterion, optimiser):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def plot_confusion_matrix(model, loader, classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            preds = model(images).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")
    print("Confusion matrix saved to outputs/confusion_matrix.png")

def main():
    print(f"Training on: {DEVICE}")
    train_tf, val_tf = get_transforms()
    train_loader, val_loader, test_loader, classes = load_data(train_tf, val_tf)
    print(f"Classes: {classes}")
    
    model     = build_model(num_classes=len(classes), device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.fc.parameters(), lr=LR)
    
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "model": "ResNet18",
            "strategy": "transfer_learning"
        })
        
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimiser)
            val_loss, val_acc     = eval_epoch(model, val_loader, criterion)
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.3f} Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.3f} Acc: {val_acc:.3f}")
        
        # Final test evaluation
        test_loss, test_acc = eval_epoch(model, test_loader, criterion)
        mlflow.log_metric("test_acc", test_acc)
        print(f"\nTest Accuracy: {test_acc:.3f}")
        
        # Save model
        os.makedirs("outputs", exist_ok=True)
        torch.save(model.state_dict(), "outputs/model.pth")
        mlflow.pytorch.log_model(model, "model")
        
        plot_confusion_matrix(model, test_loader, classes)

if __name__ == "__main__":
    main()
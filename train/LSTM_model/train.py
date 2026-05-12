"""
Training script for LSTM Dynamic Sign Model (J/Z)
"""

import sys
from pathlib import Path

# Add parent directory to path for direct execution
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from LSTM_model.model import DynamicSignLSTM, DynamicSignTransformer
from LSTM_model import config


def load_data():
    """Load and prepare data."""
    print(f"📂 Loading data from {config.DATA_PATH}")
    
    data = np.load(config.DATA_PATH, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y']
    
    # Convert labels to indices
    y_idx = np.array([config.CLASS_TO_IDX[label] for label in y])
    
    print(f"   X shape: {X.shape}")
    print(f"   Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=config.TEST_SPLIT, random_state=42, stratify=y_idx
    )
    
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, X_test, y_test


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    return total_loss / len(loader), correct / total, all_preds, all_labels


def plot_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss History')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy History')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=config.CLASSES,
        yticklabels=config.CLASSES,
        xlabel='Predicted',
        ylabel='True',
        title='Confusion Matrix'
    )
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved: {save_path}")


def train(model_type='lstm'):
    """Main training function."""
    print("=" * 60)
    print("🏋️ TRAINING DYNAMIC SIGN MODEL (J/Z)")
    print("=" * 60)
    
    device = config.DEVICE
    print(f"\n🖥️ Device: {device}")
    
    # Load data
    train_loader, test_loader, X_test, y_test = load_data()
    
    # Create model
    print(f"\n🧠 Creating {model_type.upper()} model...")
    if model_type == 'lstm':
        model = DynamicSignLSTM(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        )
    else:
        model = DynamicSignTransformer(
            input_size=config.INPUT_SIZE,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        )
    
    model = model.to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\n🚀 Training for {config.EPOCHS} epochs...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"   Epoch {epoch+1:3d}/{config.EPOCHS} | "
              f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | "
              f"Loss: {val_loss:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'config': {
                    'model_type': model_type,
                    'input_size': config.INPUT_SIZE,
                    'hidden_size': config.HIDDEN_SIZE,
                    'num_layers': config.NUM_LAYERS,
                    'num_classes': config.NUM_CLASSES
                }
            }, config.CHECKPOINT_PATH)
            patience_counter = 0
            print(f"   💾 Saved best model (acc: {val_acc:.3f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(config.CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    
    print(f"\n✅ Best Test Accuracy: {test_acc:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=config.CLASSES))
    
    # Save plots
    plot_history(history, config.MODEL_DIR / "training_history.png")
    plot_confusion_matrix(y_true, y_pred, config.MODEL_DIR / "confusion_matrix.png")
    
    # Save classes
    np.save(config.MODEL_DIR / "classes.npy", np.array(config.CLASSES))
    
    print("\n✅ Training complete!")
    print(f"   Model saved: {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    train(model_type='lstm')
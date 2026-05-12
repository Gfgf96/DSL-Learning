import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import ASLClassifier
import config


class ASLDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score, epoch):
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False

        improved = False
        if self.mode == 'max':
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = (all_preds == all_targets).mean()
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    avg_loss = total_loss / len(data_loader)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'predictions': all_preds,
        'targets': all_targets
    }


def plot_training_history(history, save_path='training_history.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # F1 Scores
    axes[1, 0].plot(history['val_macro_f1'], label='Macro F1')
    axes[1, 0].plot(history['val_weighted_f1'], label='Weighted F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Scores')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning Rate
    axes[1, 1].plot(history['learning_rates'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(targets, predictions, class_names, save_path='confusion_matrix.png'):
    cm = confusion_matrix(targets, predictions)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def train_pipeline():
    # Load data
    data = np.load(config.DATA_PATH, allow_pickle=True)
    X, y = data['X'], data['y']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    np.save(config.LABEL_ENCODER_PATH, le.classes_)
    class_names = le.classes_

    # Three-way split: Train, Val, Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded,
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        random_state=config.RANDOM_SEED,
        stratify=y_encoded
    )

    val_size_adjusted = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size_adjusted),
        random_state=config.RANDOM_SEED,
        stratify=y_temp
    )

    print(f"Dataset splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create data loaders
    train_loader = DataLoader(
        ASLDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        ASLDataset(X_val, y_val),
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    test_loader = DataLoader(
        ASLDataset(X_test, y_test),
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # Initialize model
    model = ASLClassifier(config.INPUT_SIZE, config.NUM_CLASSES).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA,
        mode='max'
    )

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_macro_f1': [], 'val_weighted_f1': [],
        'learning_rates': []
    }

    best_val_f1 = 0

    print(f"\nTraining on {config.DEVICE}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(config.EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == batch_y).sum().item()
            train_total += batch_y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, config.DEVICE)

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_weighted_f1'].append(val_metrics['weighted_f1'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Print progress
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val Macro-F1: {val_metrics['macro_f1']:.4f} | Val Weighted-F1: {val_metrics['weighted_f1']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_metrics['weighted_f1'] > best_val_f1:
            best_val_f1 = val_metrics['weighted_f1']
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  ** Best model saved! (Weighted F1: {best_val_f1:.4f}) **")

        # Learning rate scheduling
        scheduler.step(val_metrics['weighted_f1'])

        # Early stopping check
        if early_stopping(val_metrics['weighted_f1'], epoch):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"Best epoch was {early_stopping.best_epoch + 1} with Weighted F1: {early_stopping.best_score:.4f}")
            break

    # Plot training history
    plot_training_history(history)

    # Final evaluation on test set
    print("\n" + "=" * 50)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 50)

    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    test_metrics = evaluate_model(model, test_loader, criterion, config.DEVICE)

    print(f"\nTest Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Weighted-F1: {test_metrics['weighted_f1']:.4f}")

    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        test_metrics['targets'],
        test_metrics['predictions'],
        target_names=class_names,
        zero_division=0
    ))

    # Confusion matrix
    plot_confusion_matrix(
        test_metrics['targets'],
        test_metrics['predictions'],
        class_names
    )

    print(f"\nTraining complete! Best model saved to {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_pipeline()
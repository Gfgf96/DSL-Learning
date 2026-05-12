"""
CNN (MLP) model architecture for static sign language classification.
Upgraded to ResidualMLP with skip connections for better performance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connection for better gradient flow."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
    
    def forward(self, x):
        return F.gelu(x + self.block(x))


class ResidualMLP(nn.Module):
    """
    MLP with Residual connections for static sign classification.
    Better gradient flow and performance than simple MLP.
    
    Architecture:
        - Input projection: input_dim -> hidden_dim
        - Residual blocks with skip connections
        - Output classifier: hidden_dim -> num_classes
    
    Args:
        input_dim: Number of input features (default: 63 for hand landmarks)
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension (default: 256)
        num_blocks: Number of residual blocks (default: 4)
        dropout: Dropout probability (default: 0.3)
    """
    def __init__(self, input_dim=63, num_classes=24, hidden_dim=256, num_blocks=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.classifier(x)


# Legacy alias for backward compatibility
class ASLClassifier(ResidualMLP):
    """Legacy name - redirects to ResidualMLP."""
    def __init__(self, input_size, num_classes):
        super().__init__(input_dim=input_size, num_classes=num_classes)

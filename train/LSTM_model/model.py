"""
LSTM Model for Dynamic Sign Language Recognition (J/Z)
"""

import torch
import torch.nn as nn


class DynamicSignLSTM(nn.Module):
    """
    LSTM model for recognizing dynamic sign language letters.
    Input: (batch, sequence_length, features) = (batch, 30, 63)
    Output: (batch, num_classes) = (batch, 2)
    """
    
    def __init__(
        self,
        input_size=63,
        hidden_size=128,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        x: (batch, seq_len, features)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last output (or use attention)
        # lstm_out: (batch, seq_len, hidden*2)
        last_output = lstm_out[:, -1, :]  # (batch, hidden*2)
        
        # Classification
        out = self.fc(last_output)
        return out


class DynamicSignTransformer(nn.Module):
    """
    Alternative: Transformer-based model for dynamic signs.
    """
    
    def __init__(
        self,
        input_size=63,
        d_model=128,
        nhead=4,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 30, d_model) * 0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        x: (batch, seq_len, features)
        """
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        out = self.fc(x)
        return out
"""
Deep learning models for microservices workload prediction.
Includes LSTM, GRU, and Transformer-based models.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from config import MODEL_CONFIG


class LSTMPredictor(nn.Module):
    """
    LSTM-based model for time series prediction.
    Suitable for capturing long-term dependencies in workload patterns.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        num_layers: int = None,
        output_size: int = 1,
        dropout: float = None,
        bidirectional: bool = False
    ):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size or MODEL_CONFIG.lstm_hidden_size
        self.num_layers = num_layers or MODEL_CONFIG.lstm_num_layers
        self.dropout = dropout or MODEL_CONFIG.lstm_dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layer for prediction
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_directions, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output for prediction
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            out = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            out = lstm_out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        
        return out


class GRUPredictor(nn.Module):
    """
    GRU-based model for time series prediction.
    More computationally efficient than LSTM with similar performance.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        num_layers: int = None,
        output_size: int = 1,
        dropout: float = None
    ):
        super(GRUPredictor, self).__init__()
        
        self.hidden_size = hidden_size or MODEL_CONFIG.lstm_hidden_size
        self.num_layers = num_layers or MODEL_CONFIG.lstm_num_layers
        self.dropout = dropout or MODEL_CONFIG.lstm_dropout
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        gru_out, h_n = self.gru(x)
        out = gru_out[:, -1, :]
        out = self.fc(out)
        return out


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model.
    Adds position information to the input embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Transformer-based model for time series prediction.
    Uses self-attention to capture complex temporal patterns.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = None,
        nhead: int = None,
        num_layers: int = None,
        output_size: int = 1,
        dropout: float = None,
        seq_length: int = None
    ):
        super(TransformerPredictor, self).__init__()
        
        self.d_model = d_model or MODEL_CONFIG.transformer_d_model
        self.nhead = nhead or MODEL_CONFIG.transformer_nhead
        self.num_layers = num_layers or MODEL_CONFIG.transformer_num_layers
        self.dropout = dropout or MODEL_CONFIG.transformer_dropout
        self.seq_length = seq_length or MODEL_CONFIG.seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_size, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            self.d_model,
            max_len=self.seq_length,
            dropout=self.dropout
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(self.d_model * self.seq_length, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
        """
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Flatten and predict
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)
        
        return out


class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism for time series prediction.
    Combines LSTM's sequential processing with attention's ability
    to focus on important time steps.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        num_layers: int = None,
        output_size: int = 1,
        dropout: float = None
    ):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_size = hidden_size or MODEL_CONFIG.lstm_hidden_size
        self.num_layers = num_layers or MODEL_CONFIG.lstm_num_layers
        self.dropout = dropout or MODEL_CONFIG.lstm_dropout
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output
        out = self.fc(context)
        
        return out


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block.
    Uses dilated causal convolutions for sequence modeling.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super(TCNBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        """
        # First convolution
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Truncate to original length
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class TCNPredictor(nn.Module):
    """
    Temporal Convolutional Network for time series prediction.
    Alternative to LSTM with better parallelization.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_channels: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        output_size: int = 1,
        dropout: float = 0.2,
        seq_length: int = None
    ):
        super(TCNPredictor, self).__init__()
        
        self.seq_length = seq_length or MODEL_CONFIG.seq_length
        
        # Input projection
        self.input_projection = nn.Conv1d(input_size, hidden_channels, 1)
        
        # TCN blocks with increasing dilation
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(
                TCNBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )
        self.tcn = nn.Sequential(*layers)
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels * self.seq_length, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
        """
        # Transpose for Conv1d: (batch, features, seq_length)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)
        
        # TCN blocks
        x = self.tcn(x)
        
        # Flatten and predict
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)
        
        return out


class EnsemblePredictor(nn.Module):
    """
    Ensemble model combining multiple predictors.
    Uses weighted averaging for final prediction.
    """
    
    def __init__(
        self,
        models: list,
        weights: list = None
    ):
        super(EnsemblePredictor, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weighted ensemble.
        """
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Stack and weighted average
        outputs = torch.stack(outputs, dim=-1)
        weights = self.weights.to(outputs.device)
        out = torch.sum(outputs * weights, dim=-1)
        
        return out


def create_model(
    model_type: str,
    input_size: int,
    output_size: int = 1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: One of 'lstm', 'gru', 'transformer', 'attention_lstm', 'tcn'
        input_size: Number of input features
        output_size: Number of output values to predict
        **kwargs: Additional model-specific arguments
    
    Returns:
        PyTorch model
    """
    models = {
        'lstm': LSTMPredictor,
        'gru': GRUPredictor,
        'transformer': TransformerPredictor,
        'attention_lstm': AttentionLSTM,
        'tcn': TCNPredictor
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(models.keys())}")
    
    return models[model_type](
        input_size=input_size,
        output_size=output_size,
        **kwargs
    )


if __name__ == "__main__":
    # Test models
    print("Testing models...")
    
    batch_size = 32
    seq_length = 12
    input_size = 2
    output_size = 1
    
    # Create sample input
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Test each model
    for model_type in ['lstm', 'gru', 'transformer', 'attention_lstm', 'tcn']:
        print(f"\nTesting {model_type}...")
        model = create_model(model_type, input_size, output_size, seq_length=seq_length)
        
        # Forward pass
        with torch.no_grad():
            out = model(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {num_params:,}")
    
    print("\nAll models tested successfully!")



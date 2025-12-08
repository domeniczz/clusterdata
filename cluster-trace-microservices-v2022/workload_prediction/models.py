"""
Deep learning models for microservices workload prediction.
Includes LSTM, GRU, Transformer, TCN, and modern architectures like NLinear, DLinear.

References:
- NLinear/DLinear: "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
- Informer: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from config import MODEL_CONFIG


# ==================== Recurrent Models ====================

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
            out = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            out = lstm_out[:, -1, :]
        
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
        
        self.hidden_size = hidden_size or MODEL_CONFIG.gru_hidden_size
        self.num_layers = num_layers or MODEL_CONFIG.gru_num_layers
        self.dropout = dropout or MODEL_CONFIG.gru_dropout
        
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
        """Forward pass."""
        gru_out, h_n = self.gru(x)
        out = gru_out[:, -1, :]
        out = self.fc(out)
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
        """Forward pass with attention."""
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


# ==================== Transformer Models ====================

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
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
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
        
        # Ensure d_model is divisible by nhead
        if self.d_model % self.nhead != 0:
            self.d_model = (self.d_model // self.nhead + 1) * self.nhead
        
        # Input projection
        self.input_projection = nn.Linear(input_size, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            self.d_model,
            max_len=self.seq_length * 2,
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
        
        # Output layer - use last token's representation
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Use last token for prediction
        out = self.fc(x[:, -1, :])
        
        return out


# ==================== Convolutional Models ====================

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
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(out_channels)
        self.layer_norm2 = nn.LayerNorm(out_channels)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        # First convolution
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Truncate to original length
        out = out.transpose(1, 2)
        out = self.layer_norm1(out)
        out = out.transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = out.transpose(1, 2)
        out = self.layer_norm2(out)
        out = out.transpose(1, 2)
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
        hidden_channels: int = None,
        num_levels: int = None,
        kernel_size: int = None,
        output_size: int = 1,
        dropout: float = None,
        seq_length: int = None
    ):
        super(TCNPredictor, self).__init__()
        
        self.hidden_channels = hidden_channels or MODEL_CONFIG.tcn_num_channels
        self.num_levels = num_levels or MODEL_CONFIG.tcn_num_levels
        self.kernel_size = kernel_size or MODEL_CONFIG.tcn_kernel_size
        self.dropout = dropout or MODEL_CONFIG.tcn_dropout
        self.seq_length = seq_length or MODEL_CONFIG.seq_length
        
        # Input projection
        self.input_projection = nn.Conv1d(input_size, self.hidden_channels, 1)
        
        # TCN blocks with increasing dilation
        layers = []
        for i in range(self.num_levels):
            dilation = 2 ** i
            layers.append(
                TCNBlock(
                    self.hidden_channels,
                    self.hidden_channels,
                    self.kernel_size,
                    dilation,
                    self.dropout
                )
            )
        self.tcn = nn.Sequential(*layers)
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels // 2, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Transpose for Conv1d: (batch, features, seq_length)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)
        
        # TCN blocks
        x = self.tcn(x)
        
        # Use last timestep
        out = self.fc(x[:, :, -1])
        
        return out


# ==================== Linear Models (NLinear, DLinear) ====================

class NLinear(nn.Module):
    """
    NLinear: A Simple Yet Effective Baseline for Time Series Forecasting.
    Normalizes the input by the last value, applies linear layer, then denormalizes.
    
    Paper: "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
    """
    
    def __init__(
        self,
        input_size: int,
        seq_length: int = None,
        pred_length: int = 1,
        individual: bool = False
    ):
        super(NLinear, self).__init__()
        
        self.seq_length = seq_length or MODEL_CONFIG.seq_length
        self.pred_length = pred_length
        self.individual = individual
        self.input_size = input_size
        
        if self.individual:
            # Individual linear layer for each feature
            self.linear = nn.ModuleList([
                nn.Linear(self.seq_length, self.pred_length) 
                for _ in range(input_size)
            ])
        else:
            # Shared linear layer
            self.linear = nn.Linear(self.seq_length, self.pred_length)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (batch_size, seq_length, input_size)
        Returns:
            (batch_size, pred_length) - only the target feature
        """
        # Get the last value for normalization
        seq_last = x[:, -1:, :].detach()
        
        # Normalize
        x = x - seq_last
        
        if self.individual:
            outputs = []
            for i, linear in enumerate(self.linear):
                out = linear(x[:, :, i])  # (batch, pred_length)
                outputs.append(out)
            x = torch.stack(outputs, dim=-1)  # (batch, pred_length, input_size)
        else:
            x = x.transpose(1, 2)  # (batch, input_size, seq_length)
            x = self.linear(x)     # (batch, input_size, pred_length)
            x = x.transpose(1, 2)  # (batch, pred_length, input_size)
        
        # Denormalize
        x = x + seq_last
        
        # Return only the target feature (first feature)
        return x[:, :, 0].squeeze(-1)


class MovingAvgBlock(nn.Module):
    """Moving average block for trend extraction."""
    
    def __init__(self, kernel_size: int):
        super(MovingAvgBlock, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply moving average with padding."""
        # x: (batch, seq_length, features)
        # Pad to maintain length
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        
        # Apply average pooling
        x = x.transpose(1, 2)  # (batch, features, seq_length)
        x = self.avg(x)
        x = x.transpose(1, 2)  # (batch, seq_length, features)
        
        return x


class SeriesDecomp(nn.Module):
    """Series decomposition block for DLinear."""
    
    def __init__(self, kernel_size: int = 25):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvgBlock(kernel_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose series into trend and seasonal components."""
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class DLinear(nn.Module):
    """
    DLinear: Decomposition Linear Model for Time Series Forecasting.
    Decomposes time series into trend and seasonal, applies linear to each.
    
    Paper: "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
    """
    
    def __init__(
        self,
        input_size: int,
        seq_length: int = None,
        pred_length: int = 1,
        individual: bool = False,
        kernel_size: int = 25
    ):
        super(DLinear, self).__init__()
        
        self.seq_length = seq_length or MODEL_CONFIG.seq_length
        self.pred_length = pred_length
        self.individual = individual
        self.input_size = input_size
        
        # Series decomposition
        self.decomposition = SeriesDecomp(kernel_size)
        
        if self.individual:
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(self.seq_length, self.pred_length) 
                for _ in range(input_size)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(self.seq_length, self.pred_length) 
                for _ in range(input_size)
            ])
        else:
            self.linear_seasonal = nn.Linear(self.seq_length, self.pred_length)
            self.linear_trend = nn.Linear(self.seq_length, self.pred_length)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Decompose
        seasonal, trend = self.decomposition(x)
        
        if self.individual:
            seasonal_outputs = []
            trend_outputs = []
            for i in range(self.input_size):
                seasonal_outputs.append(self.linear_seasonal[i](seasonal[:, :, i]))
                trend_outputs.append(self.linear_trend[i](trend[:, :, i]))
            seasonal_out = torch.stack(seasonal_outputs, dim=-1)
            trend_out = torch.stack(trend_outputs, dim=-1)
        else:
            seasonal = seasonal.transpose(1, 2)
            trend = trend.transpose(1, 2)
            seasonal_out = self.linear_seasonal(seasonal).transpose(1, 2)
            trend_out = self.linear_trend(trend).transpose(1, 2)
        
        x = seasonal_out + trend_out
        
        # Return only the target feature
        return x[:, :, 0].squeeze(-1)


# ==================== Ensemble Models ====================

class EnsemblePredictor(nn.Module):
    """
    Ensemble model combining multiple predictors.
    Uses weighted averaging for final prediction.
    """
    
    def __init__(self, models: list, weights: list = None):
        super(EnsemblePredictor, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.register_buffer('weights', torch.tensor(weights))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weighted ensemble."""
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Stack and weighted average
        outputs = torch.stack(outputs, dim=-1)
        out = torch.sum(outputs * self.weights, dim=-1)
        
        return out


# ==================== Model Factory ====================

def create_model(
    model_type: str,
    input_size: int,
    output_size: int = 1,
    seq_length: int = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: One of 'lstm', 'gru', 'transformer', 'attention_lstm', 
                   'tcn', 'nlinear', 'dlinear'
        input_size: Number of input features
        output_size: Number of output values to predict
        seq_length: Sequence length (required for some models)
        **kwargs: Additional model-specific arguments
    
    Returns:
        PyTorch model
    """
    if seq_length is None:
        seq_length = MODEL_CONFIG.seq_length
    
    models = {
        'lstm': LSTMPredictor,
        'gru': GRUPredictor,
        'transformer': TransformerPredictor,
        'attention_lstm': AttentionLSTM,
        'tcn': TCNPredictor,
        'nlinear': NLinear,
        'dlinear': DLinear
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(models.keys())}")
    
    # Handle different model signatures
    if model_type in ['nlinear', 'dlinear']:
        return models[model_type](
            input_size=input_size,
            seq_length=seq_length,
            pred_length=output_size,
            **kwargs
        )
    elif model_type in ['transformer', 'tcn']:
        return models[model_type](
            input_size=input_size,
            output_size=output_size,
            seq_length=seq_length,
            **kwargs
        )
    else:
        return models[model_type](
            input_size=input_size,
            output_size=output_size,
            **kwargs
        )


def get_model_info(model: nn.Module) -> dict:
    """Get model information including parameter count."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_all_params = sum(p.numel() for p in model.parameters())
    
    return {
        'trainable_params': num_params,
        'total_params': num_all_params,
        'model_class': model.__class__.__name__
    }


if __name__ == "__main__":
    # Test models
    print("Testing deep learning models...")
    
    batch_size = 32
    seq_length = 24
    input_size = 2
    output_size = 1
    
    # Create sample input
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Test each model
    model_types = ['lstm', 'gru', 'transformer', 'attention_lstm', 'tcn', 'nlinear', 'dlinear']
    
    for model_type in model_types:
        print(f"\nTesting {model_type}...")
        model = create_model(model_type, input_size, output_size, seq_length=seq_length)
        
        # Forward pass
        with torch.no_grad():
            out = model(x)
        
        info = get_model_info(model)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Parameters: {info['trainable_params']:,}")
    
    print("\nAll models tested successfully!")

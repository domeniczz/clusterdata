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
        
        # Input projection for residual connection
        self.input_proj = nn.Linear(input_size, self.hidden_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Scaled dot-product attention (more stable than additive attention)
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.scale = math.sqrt(self.hidden_size)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        
        # Output layer with residual
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, output_size)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot for better training."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with scaled dot-product attention."""
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Scaled dot-product attention
        # Use last hidden state as query
        query = self.query(lstm_out[:, -1:, :])  # (batch, 1, hidden)
        key = self.key(lstm_out)                  # (batch, seq, hidden)
        value = self.value(lstm_out)              # (batch, seq, hidden)
        
        # Attention scores
        attn_scores = torch.bmm(query, key.transpose(1, 2)) / self.scale  # (batch, 1, seq)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Context vector
        context = torch.bmm(attn_weights, value)  # (batch, 1, hidden)
        context = context.squeeze(1)               # (batch, hidden)
        
        # Residual connection: add last LSTM output
        context = context + lstm_out[:, -1, :]
        
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
            cos_positions = min(d_model // 2, len(div_term))
            pe[:, 1::2] = torch.cos(position * div_term[:cos_positions])
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
        out = x[:, :, 0]  # (batch, pred_length)
        # Squeeze only if pred_length == 1 to get (batch,)
        if self.pred_length == 1:
            return out.squeeze(-1)
        return out


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


# ==================== Advanced Models ====================

class PatchTST(nn.Module):
    """
    PatchTST: A Time Series is Worth 64 Words
    
    Patches the time series and applies transformer encoder.
    Reference: ICLR 2023 - "A Time Series is Worth 64 Words: Long-term 
    Forecasting with Transformers"
    """
    
    def __init__(
        self,
        input_size: int,
        seq_length: int = None,
        pred_length: int = 1,
        patch_len: int = 4,
        stride: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(PatchTST, self).__init__()
        
        self.input_size = input_size
        self.seq_length = seq_length or MODEL_CONFIG.seq_length
        self.pred_length = pred_length
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # Calculate number of patches
        self.num_patches = max(1, (self.seq_length - patch_len) // stride + 1)
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_len * input_size, d_model)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model * self.num_patches, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_length)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_length, input_size)
        Returns:
            (batch, pred_length)
        """
        batch_size = x.shape[0]
        
        # Create patches
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            if end <= x.shape[1]:
                patch = x[:, start:end, :].reshape(batch_size, -1)
                patches.append(patch)
        
        if not patches:
            # Fallback: use entire sequence as one patch
            patches = [x.reshape(batch_size, -1)[:, :self.patch_len * self.input_size]]
        
        # Stack patches
        patches = torch.stack(patches, dim=1)  # (batch, num_patches, patch_len * input_size)
        
        # Patch embedding
        x = self.patch_embed(patches)  # (batch, num_patches, d_model)
        
        # Add positional encoding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        # Flatten and predict
        x = x.reshape(batch_size, -1)
        out = self.head(x)
        
        return out.squeeze(-1) if self.pred_length == 1 else out


class Informer(nn.Module):
    """
    Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
    
    Simplified version focusing on ProbSparse attention.
    Reference: AAAI 2021
    """
    
    def __init__(
        self,
        input_size: int,
        seq_length: int = None,
        pred_length: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        factor: int = 5
    ):
        super(Informer, self).__init__()
        
        self.seq_length = seq_length or MODEL_CONFIG.seq_length
        self.pred_length = pred_length
        self.d_model = d_model
        self.factor = factor
        
        # Input embedding
        self.input_embed = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_length * 2, dropout=dropout)
        
        # Encoder layers with ProbSparse attention approximation
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, nhead, d_model * 4, dropout, factor)
            for _ in range(num_layers)
        ])
        
        # Distilling layers for sequence reduction
        self.distilling = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
            for _ in range(num_layers - 1)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * max(1, self.seq_length // (2 ** (num_layers - 1))), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_length)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input embedding
        x = self.input_embed(x)
        x = self.pos_encoder(x)
        
        # Encoder with distilling
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if i < len(self.distilling):
                x = x.transpose(1, 2)
                x = self.distilling[i](x)
                x = x.transpose(1, 2)
        
        # Output
        out = self.output_proj(x)
        return out.squeeze(-1) if self.pred_length == 1 else out


class InformerEncoderLayer(nn.Module):
    """Informer encoder layer with approximated ProbSparse attention."""
    
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float, factor: int):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class TimesBlock(nn.Module):
    """
    TimesBlock from TimesNet.
    Converts 1D time series to 2D and applies inception-like convolutions.
    """
    
    def __init__(self, d_model: int, d_ff: int, num_kernels: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_kernels = num_kernels
        
        # 2D convolution for period-wise processing
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_ff, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(d_ff, d_model, kernel_size=(1, 3), padding=(0, 1))
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, period: int = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            period: detected period for reshaping
        """
        batch, seq_len, d_model = x.shape
        
        # Detect period using FFT if not provided
        if period is None:
            period = max(2, seq_len // 4)
        
        # Pad sequence to be divisible by period
        if seq_len % period != 0:
            pad_len = period - (seq_len % period)
            x = F.pad(x, (0, 0, 0, pad_len))
            seq_len = x.shape[1]
        
        # Reshape to 2D: (batch, d_model, period, seq_len//period)
        x_2d = x.transpose(1, 2).reshape(batch, d_model, period, seq_len // period)
        
        # 2D convolution
        out_2d = self.conv(x_2d)
        
        # Reshape back to 1D
        out = out_2d.reshape(batch, d_model, -1).transpose(1, 2)
        
        # Trim to original length
        out = out[:, :seq_len, :]
        
        return self.dropout(out)


class TimesNet(nn.Module):
    """
    TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
    
    Reference: ICLR 2023
    """
    
    def __init__(
        self,
        input_size: int,
        seq_length: int = None,
        pred_length: int = 1,
        d_model: int = 64,
        d_ff: int = 128,
        num_layers: int = 2,
        num_kernels: int = 6,
        dropout: float = 0.1
    ):
        super(TimesNet, self).__init__()
        
        self.seq_length = seq_length or MODEL_CONFIG.seq_length
        self.pred_length = pred_length
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Input embedding
        self.input_embed = nn.Linear(input_size, d_model)
        
        # TimesBlocks
        self.blocks = nn.ModuleList([
            TimesBlock(d_model, d_ff, num_kernels, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * self.seq_length, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_length)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input embedding
        x = self.input_embed(x)
        
        # TimesBlocks with residual
        for block, norm in zip(self.blocks, self.norms):
            residual = x
            x = block(x)
            x = norm(x + residual)
        
        # Prediction
        out = self.head(x)
        return out.squeeze(-1) if self.pred_length == 1 else out


class Autoformer(nn.Module):
    """
    Autoformer: Decomposition Transformers with Auto-Correlation
    
    Simplified version with series decomposition and auto-correlation.
    Reference: NeurIPS 2021
    """
    
    def __init__(
        self,
        input_size: int,
        seq_length: int = None,
        pred_length: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        moving_avg: int = 25
    ):
        super(Autoformer, self).__init__()
        
        self.seq_length = seq_length or MODEL_CONFIG.seq_length
        self.pred_length = pred_length
        self.d_model = d_model
        
        # Series decomposition
        self.decomp = SeriesDecomp(kernel_size=moving_avg)
        
        # Input embedding
        self.input_embed = nn.Linear(input_size, d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, nhead, d_model * 4, dropout, moving_avg)
            for _ in range(num_layers)
        ])
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model * self.seq_length, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_length)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Decomposition
        seasonal, trend = self.decomp(x)
        
        # Encode seasonal component
        x = self.input_embed(seasonal)
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Flatten and predict
        x = x.reshape(x.shape[0], -1)
        out = self.head(x)
        
        return out.squeeze(-1) if self.pred_length == 1 else out


class AutoformerEncoderLayer(nn.Module):
    """Autoformer encoder layer with auto-correlation attention."""
    
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float, moving_avg: int):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Auto-correlation attention
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        
        # First decomposition
        seasonal1, trend1 = self.decomp1(x)
        
        # Feed forward
        ff_out = self.feed_forward(seasonal1)
        seasonal2 = seasonal1 + self.dropout(ff_out)
        
        # Second decomposition
        seasonal_out, trend2 = self.decomp2(seasonal2)
        
        return seasonal_out


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
                   'tcn', 'nlinear', 'dlinear', 'patchtst', 'informer',
                   'timesnet', 'autoformer'
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
        # Classic models
        'lstm': LSTMPredictor,
        'gru': GRUPredictor,
        'attention_lstm': AttentionLSTM,
        
        # Transformer-based models
        'transformer': TransformerPredictor,
        'patchtst': PatchTST,
        'informer': Informer,
        'autoformer': Autoformer,
        
        # CNN-based models
        'tcn': TCNPredictor,
        'timesnet': TimesNet,
        
        # Linear models
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
    elif model_type in ['patchtst', 'informer', 'timesnet', 'autoformer']:
        return models[model_type](
            input_size=input_size,
            seq_length=seq_length,
            pred_length=output_size,
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


def get_available_models() -> List[str]:
    """Get list of all available model types."""
    return [
        'lstm', 'gru', 'attention_lstm',  # RNN-based
        'transformer', 'patchtst', 'informer', 'autoformer',  # Transformer-based
        'tcn', 'timesnet',  # CNN-based
        'nlinear', 'dlinear'  # Linear
    ]


def get_model_summary() -> dict:
    """Get a summary of all available models with their characteristics."""
    return {
        'lstm': {'type': 'RNN', 'complexity': 'Medium', 'best_for': 'Short sequences'},
        'gru': {'type': 'RNN', 'complexity': 'Medium', 'best_for': 'Short sequences, faster than LSTM'},
        'attention_lstm': {'type': 'RNN+Attention', 'complexity': 'Medium', 'best_for': 'Variable importance sequences'},
        'transformer': {'type': 'Transformer', 'complexity': 'High', 'best_for': 'Complex patterns'},
        'patchtst': {'type': 'Transformer', 'complexity': 'Medium', 'best_for': 'Long sequences'},
        'informer': {'type': 'Transformer', 'complexity': 'High', 'best_for': 'Very long sequences'},
        'autoformer': {'type': 'Transformer', 'complexity': 'High', 'best_for': 'Seasonal patterns'},
        'tcn': {'type': 'CNN', 'complexity': 'Medium', 'best_for': 'Parallel processing'},
        'timesnet': {'type': 'CNN', 'complexity': 'High', 'best_for': 'Multi-period patterns'},
        'nlinear': {'type': 'Linear', 'complexity': 'Low', 'best_for': 'Simple patterns, baseline'},
        'dlinear': {'type': 'Linear', 'complexity': 'Low', 'best_for': 'Trend + seasonal decomposition'}
    }


if __name__ == "__main__":
    # Test models
    print("=" * 60)
    print("TESTING ALL DEEP LEARNING MODELS")
    print("=" * 60)
    
    batch_size = 32
    seq_length = 24
    input_size = 2
    output_size = 1
    
    # Create sample input
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Test each model
    model_types = get_available_models()
    
    print(f"\nTesting {len(model_types)} models with:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Input features: {input_size}")
    print(f"  Output size: {output_size}")
    
    results = []
    
    for model_type in model_types:
        print(f"\n{'='*40}")
        print(f"Testing {model_type.upper()}...")
        
        try:
            model = create_model(model_type, input_size, output_size, seq_length=seq_length)
            
            # Forward pass
            with torch.no_grad():
                out = model(x)
            
            info = get_model_info(model)
            
            print(f"  ✓ Input shape: {x.shape}")
            print(f"  ✓ Output shape: {out.shape}")
            print(f"  ✓ Parameters: {info['trainable_params']:,}")
            
            results.append({
                'model': model_type,
                'status': 'OK',
                'params': info['trainable_params'],
                'output_shape': tuple(out.shape)
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'model': model_type,
                'status': 'FAILED',
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r['status'] == 'OK')
    failed = len(results) - passed
    
    print(f"\nPassed: {passed}/{len(results)}")
    if failed > 0:
        print(f"Failed: {failed}")
        for r in results:
            if r['status'] == 'FAILED':
                print(f"  - {r['model']}: {r.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("All model tests completed!")

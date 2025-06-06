import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class PriceActionAttention(nn.Module):
    """Custom attention mechanism for price action patterns."""
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Pattern-specific attention weights
        self.pattern_weights = nn.Parameter(torch.ones(n_heads, 1) / n_heads)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply pattern-specific attention weights
        pattern_attention = F.softmax(self.pattern_weights, dim=0)
        scores = scores * pattern_attention.view(1, self.n_heads, 1, 1)
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, v)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(context)

class ReasoningLayer(nn.Module):
    """Layer that implements reasoning about price action patterns and market structure."""
    def __init__(self, d_model: int, n_patterns: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_patterns = n_patterns
        
        # Pattern recognition
        self.pattern_encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, n_patterns)
        )
        
        # Structure analysis
        self.structure_encoder = nn.Sequential(
            nn.Linear(d_model + n_patterns, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # Reasoning gates
        self.reasoning_gates = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pattern recognition
        patterns = self.pattern_encoder(x)
        pattern_probs = F.softmax(patterns, dim=-1)
        
        # Combine with original features
        combined = torch.cat([x, pattern_probs], dim=-1)
        
        # Structure analysis
        structure = self.structure_encoder(combined)
        
        # Reasoning gates
        gates = self.reasoning_gates(torch.cat([x, structure], dim=-1))
        
        # Apply reasoning
        reasoned = gates * structure + (1 - gates) * x
        
        return reasoned, pattern_probs

class PriceActionTransformer(nn.Module):
    """Custom transformer model for price action analysis."""
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': PriceActionAttention(d_model, n_heads),
                'norm1': nn.LayerNorm(d_model),
                'reasoning': ReasoningLayer(d_model),
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model)
                ),
                'norm3': nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])
        
        # Output layers
        self.pattern_classifier = nn.Linear(d_model, 8)  # 8 pattern types
        self.direction_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # [UP, DOWN, NEUTRAL]
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len]
            
        Returns:
            Tuple of (direction_logits, pattern_logits, confidence, attention_weights)
        """
        # Input projection and positional encoding
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # Store attention weights for analysis
        attention_weights = []
        
        # Transformer layers
        for layer in self.transformer_layers:
            # Self-attention
            attn_output = layer['attention'](x, mask)
            x = layer['norm1'](x + attn_output)
            
            # Reasoning
            reasoned, pattern_probs = layer['reasoning'](x)
            x = layer['norm2'](x + reasoned)
            
            # Feed-forward
            ffn_output = layer['ffn'](x)
            x = layer['norm3'](x + ffn_output)
            
            # Store attention weights
            attention_weights.append(attn_output)
        
        # Get final sequence representation (use last token)
        final_repr = x[:, -1, :]
        
        # Direction prediction
        direction_logits = self.direction_classifier(final_repr)
        
        # Pattern classification
        pattern_logits = self.pattern_classifier(final_repr)
        
        # Confidence estimation
        confidence = self.confidence_estimator(final_repr)
        
        return direction_logits, pattern_logits, confidence, attention_weights
    
    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence thresholding.
        
        Args:
            x: Input tensor
            threshold: Confidence threshold for predictions
            
        Returns:
            Tuple of (direction, confidence, patterns)
        """
        self.eval()
        with torch.no_grad():
            direction_logits, pattern_logits, confidence, _ = self(x)
            
            # Get direction probabilities
            direction_probs = F.softmax(direction_logits, dim=-1)
            pattern_probs = F.softmax(pattern_logits, dim=-1)
            
            # Apply confidence threshold
            mask = confidence >= threshold
            direction_probs = direction_probs * mask
            
            # Get final predictions
            direction = torch.argmax(direction_probs, dim=-1)
            patterns = torch.argmax(pattern_probs, dim=-1)
            
            return direction, confidence, patterns
    
    def explain_prediction(
        self,
        x: torch.Tensor,
        attention_weights: List[torch.Tensor]
    ) -> str:
        """
        Generate a human-readable explanation of the model's prediction.
        
        Args:
            x: Input tensor
            attention_weights: List of attention weights from forward pass
            
        Returns:
            String explanation of the prediction
        """
        direction_logits, pattern_logits, confidence, _ = self(x)
        direction_probs = F.softmax(direction_logits, dim=-1)
        pattern_probs = F.softmax(pattern_logits, dim=-1)
        
        # Get top patterns
        top_patterns = torch.topk(pattern_probs, k=3)
        
        # Analyze attention weights
        last_layer_attention = attention_weights[-1]
        important_candles = torch.topk(last_layer_attention.mean(dim=1), k=3)
        
        # Generate explanation
        direction = torch.argmax(direction_probs, dim=-1).item()
        conf = confidence.item()
        
        directions = ["DOWN", "NEUTRAL", "UP"]
        patterns = [
            "Doji", "Engulfing Bullish", "Engulfing Bearish",
            "Pin Bar Bullish", "Pin Bar Bearish", "Inside Bar",
            "Outside Bar", "Break of Structure"
        ]
        
        explanation = f"Model predicts {directions[direction]} with {conf:.1%} confidence.\n"
        explanation += "Key patterns detected:\n"
        
        for i, (pattern_idx, prob) in enumerate(zip(top_patterns.indices[0], top_patterns.values[0])):
            explanation += f"- {patterns[pattern_idx]}: {prob:.1%}\n"
            
        explanation += "\nMost influential candles:\n"
        for i, (candle_idx, weight) in enumerate(zip(important_candles.indices[0], important_candles.values[0])):
            explanation += f"- Candle {candle_idx}: {weight:.1%} influence\n"
            
        return explanation 
"""
model.py

Defines a Transformer‐based language model from scratch, using PyTorch’s nn.Transformer.
No pretrained weights—everything is initialized randomly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def top_k_logits(logits: torch.Tensor, k: int):
    """
    Zero out all but the top k logits in each row; return modified logits.
    logits: (vocab_size,)
    """
    if k == 0:
        return logits
    topk_vals, _ = torch.topk(logits, k)
    min_topk = topk_vals[-1]
    return torch.where(logits < min_topk, torch.full_like(logits, -1e10), logits)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_length, d_model)
        returns x + positional encodings for the first seq_length positions.
        """
        x = x + self.pe[:, : x.size(1), :]
        return x


class NoraTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.model_type = "TransformerLM"
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding + positional encoding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        # Final linear layer to project to vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.zeros_(self.fc_out.bias)
        nn.init.normal_(self.fc_out.weight, mean=0, std=self.d_model ** -0.5)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src: (batch_size, seq_length), token IDs
        returns: logits (batch_size, seq_length, vocab_size)
        """

        # Embed tokens and add positional encoding
        x = self.token_embed(src) * math.sqrt(self.d_model)  # (B, S, D)
        x = self.pos_encoder(x)  # (B, S, D)
        # PyTorch Transformer expects (S, B, D)
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, d_model)

        # Create a causal mask so each position can only attend to previous positions
        seq_len = x.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # Pass through Transformer encoder
        x = self.transformer_encoder(x, mask=mask)  # (seq_length, batch_size, d_model)

        # Back to (B, S, D)
        x = x.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        logits = self.fc_out(x)  # (batch_size, seq_length, vocab_size)
        return logits

    def generate(
        self,
        tokenizer,
        device: str,
        prompt: str,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str:
        """
        Autoregressively generate text from a prompt.
        - tokenizer: CharTokenizer (for encode/decode)
        - device: "cuda" or "cpu"
        - prompt: initial string
        - max_length: total tokens to generate (including prompt)
        - temperature: scales logits before softmax
        - top_k: keep only top_k logits at each step
        """
        self.eval()
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        generated = input_ids.clone()  # shape (1, seq_len)
        for _ in range(max_length - input_ids.size(1)):
            # 1) trim to last seq_length tokens if longer than context window
            if generated.size(1) > self.pos_encoder.pe.size(1):
                generated = generated[:, -self.pos_encoder.pe.size(1) :]

            with torch.no_grad():
                logits = self.forward(generated)  # (1, seq_len, vocab_size)
                next_token_logits = logits[0, -1, :] / temperature
                filtered_logits = top_k_logits(next_token_logits, k=top_k)
                probs = F.softmax(filtered_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # (1,)
                generated = torch.cat([generated, next_id.unsqueeze(0)], dim=1)

        output_ids = generated.squeeze(0).tolist()
        return tokenizer.decode(output_ids)
import torch
import torch.nn as nn


class ARMovieLM(nn.Module):
    """
    Small autoregressive Transformer for next-item recommendation.

    MovieLens item IDs are expected to be remapped to contiguous
    token IDs before being passed to the model.

    Token ID 0 is reserved for padding.
    """

    def __init__(
        self,
        num_items,
        hidden_size=32,
        n_layers=1,
        n_heads=2,
        ff_size=128,
        max_seq_length=10,
        dropout=0.1,
    ):
        super().__init__()

        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length

        self.item_embedding = nn.Embedding(
            num_items + 1,
            hidden_size,
            padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            max_seq_length,
            hidden_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

        self.output = nn.Linear(
            hidden_size,
            num_items + 1,
            bias=False,
        )

        self.output.weight = self.item_embedding.weight

    def forward(self, input_ids):

        batch_size, seq_length = input_ids.shape

        positions = torch.arange(
            seq_length,
            device=input_ids.device,
        )

        positions = positions.unsqueeze(0).expand(
            batch_size,
            seq_length,
        )

        x = (
            self.item_embedding(input_ids)
            + self.position_embedding(positions)
        )

        padding_mask = input_ids.eq(0)

        causal_mask = torch.triu(
            torch.ones(
                seq_length,
                seq_length,
                device=input_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        x = self.norm(x)

        return self.output(x)

    def predict_next(self, input_ids):

        logits = self.forward(input_ids)

        return logits[:, -1, :]
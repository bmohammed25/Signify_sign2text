"""
training/model_b.py
Model B — LSTM sequence classifier for ASL word recognition.

Input:  (batch, 30, 1629)  — 30 frames × 543 landmarks × 3 coords
Output: (batch, 2410)      — logit per class

Architecture
------------
  Input (B, T, 1629)
       │
  LayerNorm  ← stabilises wildly-scaled landmark coordinates
       │
  Linear projection 1629 → 512   ← reduce feature dim before LSTM
       │
  LSTM  512 hidden, 3 layers, dropout=0.3 between layers
       │   (returns all hidden states + final h_n)
       │
  Last hidden state  h[:, -1, :]   (B, 512)
       │
  BN → Dropout(0.4) → Linear(512, 256) → ReLU
       │
  BN → Dropout(0.3) → Linear(256, num_classes)
       │
  Raw logits  (B, 2410)

Design notes
------------
- LayerNorm on raw input prevents exploding gradients from face coords
  (MediaPipe face landmarks have much larger absolute values than hands)
- Linear projection before LSTM keeps VRAM < 2 GB for batch_size=16
- 3 LSTM layers with inter-layer dropout(0.3) — good tradeoff for 4 GB GPU
- BN + Dropout in classifier head — reduces overfit on long-tail classes
- Weights init with orthogonal (LSTM) + kaiming (linear) for fast convergence
"""

import torch
import torch.nn as nn


class SignifyLSTM(nn.Module):
    """
    LSTM-based ASL word recogniser.

    Parameters
    ----------
    input_size  : int   — feature dim per frame (default 1629)
    hidden_size : int   — LSTM hidden units per layer (default 512)
    num_layers  : int   — stacked LSTM layers (default 3)
    num_classes : int   — output vocabulary size (default 2410)
    proj_size   : int   — linear projection dim before LSTM (default 512)
    lstm_dropout: float — dropout between LSTM layers (default 0.3)
    fc_dropout  : float — dropout in classifier head (default 0.4)
    """

    def __init__(
        self,
        input_size: int = 1629,
        hidden_size: int = 512,
        num_layers: int = 3,
        num_classes: int = 2410,
        proj_size: int = 512,
        lstm_dropout: float = 0.3,
        fc_dropout: float = 0.4,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ── input normalisation ─────────────────────────────────────────
        self.input_norm = nn.LayerNorm(input_size)

        # ── feature projection ──────────────────────────────────────────
        # Reduces 1629 → proj_size before the LSTM to save VRAM
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, proj_size),
            nn.ReLU(),
        )

        # ── sequence encoder ────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=proj_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            bidirectional=False,   # unidirectional — keep param count low
        )

        # ── classifier head ─────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(fc_dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(fc_dropout - 0.1),   # 0.3
            nn.Linear(256, num_classes),
        )

        # ── weight init ─────────────────────────────────────────────────
        self._init_weights()

    # ────────────────────────────────────────────────────────────────────
    def _init_weights(self):
        """Orthogonal init for LSTM; Kaiming for linear layers."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_normal_(param, nonlinearity="relu")
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1 — helps long sequences
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_size)  float32

        Returns
        -------
        logits : (B, num_classes)
        """
        # Normalise input per-frame
        x = self.input_norm(x)                  # (B, T, 1629)

        # Project to smaller feature space
        B, T, _ = x.shape
        x = x.view(B * T, -1)                   # (B*T, 1629)
        x = self.input_proj(x)                  # (B*T, 512)
        x = x.view(B, T, -1)                    # (B, T, 512)

        # Run LSTM
        out, (h_n, _) = self.lstm(x)            # out: (B, T, 512)

        # Take last hidden state of top layer
        feat = h_n[-1]                          # (B, 512)

        # Classify
        logits = self.classifier(feat)          # (B, 2410)
        return logits

    # ────────────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── convenience factory ─────────────────────────────────────────────────
def build_model(num_classes: int = 2410, device: str = "cuda") -> SignifyLSTM:
    """Build and move model to device. Prints param count."""
    model = SignifyLSTM(num_classes=num_classes)
    model = model.to(device)
    n = model.count_parameters()
    print(f"[Model B] SignifyLSTM — {n:,} trainable parameters")
    print(f"[Model B] Device: {device}")
    return model


# ── smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(num_classes=2410, device=device)

    # Dummy batch: 4 clips, 30 frames, 1629 features
    dummy = torch.randn(4, 30, 1629).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"[Smoke test] Input: {dummy.shape}  →  Output: {out.shape}")
    assert out.shape == (4, 2410), "Shape mismatch!"
    print("[Smoke test] PASSED ✅")

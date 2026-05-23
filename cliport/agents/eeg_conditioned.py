"""EEG-conditioned CLIPort agent (skeleton).

Wraps `TwoStreamClipLingUNetLatTransporterAgent` (the published "cliport"
agent) and adds an EEG embedding branch. The EEG window is the same
goal-shown -> place-done window that the two-head EEGNet was trained on
(see `training/train_clport_eegnet.py`).

Integration approach (single point of change):

    text_token   = clip_text_encoder(lang_goal)         # (B, D_text)
    eeg_embed    = eeg_encoder(eeg_window)              # (B, D_eeg)
    fused_token  = text_token + eeg_proj(eeg_embed)     # (B, D_text)

`fused_token` is then handed to LingUNet's language fusion path in place of
the original `text_token`. `eeg_proj` is a learned linear projection from
`D_eeg` -> `D_text`.

Status: SKELETON. The `_inject_fused_token` hook below shows where to
splice the EEG embedding into the parent's text-conditioning pipeline,
but the upstream `TwoStreamAttentionLangFusion` /
`TwoStreamTransportLangFusion` modules currently re-encode `lang_goal`
inside their forward pass from a *string*, which means a clean injection
needs a small upstream patch. Two viable options:

  (A) Patch upstream to accept a precomputed text embedding override; pass
      `fused_token` as that override.
  (B) Concatenate the EEG embedding into the lang_goal *string* via a
      special token that the CLIP tokenizer encodes; the model then has
      to learn to decode it. Hacky; less recommended.

Until either (A) or (B) is wired, this class falls through to the parent
behavior (i.e., it ignores EEG). It's still useful: registering it lets
end-to-end scripts run without errors while we iterate on the integration.
"""

import torch
import torch.nn as nn

from cliport.agents.transporter_lang_goal import (
    TwoStreamClipLingUNetLatTransporterAgent,
)


class EEGEncoder(nn.Module):
    """EEGNet trunk wrapper with an output projection.

    Mirrors `training.train_clport_eegnet.EEGNetTrunk` so the trunk weights
    are loadable directly via `state_dict`.
    """

    def __init__(self, n_channels: int = 8, n_samples: int = 2048,
                 out_dim: int = 512, dropout: float = 0.25):
        super().__init__()
        F1, D, F2 = 8, 2, 16
        kernel_t = 64
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_t), padding=(0, kernel_t // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flat = self.separable(self.depthwise(self.block1(dummy))).numel()
        self.flat_dim = flat
        self.proj = nn.Linear(flat, out_dim)

    def load_two_head_trunk(self, state_dict):
        """Initialize block1/depthwise/separable from a two-head checkpoint.

        Accepts either a raw EEGNetTrunk state_dict or a full
        EEGNetTwoHead state_dict (in which case `trunk.*` keys are mapped).
        """
        own = self.state_dict()
        renamed = {}
        for k, v in state_dict.items():
            key = k[len("trunk."):] if k.startswith("trunk.") else k
            if key in own and own[key].shape == v.shape:
                renamed[key] = v
        missing = [k for k in own if k not in renamed and not k.startswith("proj.")]
        self.load_state_dict({**own, **renamed}, strict=False)
        return missing

    def forward(self, x):
        # x: (B, 8, T) or (B, 1, 8, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class TwoStreamClipLingUNetLatTransporterEEGAgent(
        TwoStreamClipLingUNetLatTransporterAgent):
    """CLIPort + EEG embedding fused into the language token.

    SKELETON: see module docstring. The EEG encoder is constructed and
    optionally initialized from a pretrained two-head EEGNet checkpoint,
    but the fused token is not yet wired into the upstream language
    fusion modules. PRs welcome.
    """

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)
        eeg_T = int(cfg.get("eeg_T", 2048))
        eeg_dim = int(cfg.get("eeg_dim", 512))
        self.eeg_encoder = EEGEncoder(
            n_channels=8, n_samples=eeg_T, out_dim=eeg_dim,
        )
        ckpt = cfg.get("eeg_pretrained", None)
        if ckpt:
            sd = torch.load(ckpt, map_location="cpu")
            missing = self.eeg_encoder.load_two_head_trunk(sd)
            print(f"[eeg-agent] loaded EEG trunk from {ckpt} "
                  f"(missing keys: {missing})")

    # The parent class's `attn_training_step` / `transport_training_step`
    # consume `frame['lang_goal']`. The HITL RavensDataset emits
    # `eeg.pkl` per episode (see training/clport/build_ravens_dataset.py).
    # To wire the EEG path:
    #   1. Override `RavensDataset.__getitem__` upstream (or in a
    #      subclass) to also return `frame['eeg']`.
    #   2. Override `attn_training_step` here to compute
    #      `eeg_embed = self.eeg_encoder(frame['eeg'])` and pass it
    #      forward.
    #   3. Patch `TwoStreamAttentionLangFusion.forward` to accept an
    #      optional `text_emb_override` argument; supply
    #      `fused_token = text_emb + eeg_proj(eeg_embed)`.
    # Each step is local; the EEG path is fully optional.

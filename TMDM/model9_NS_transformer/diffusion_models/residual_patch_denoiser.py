import torch
import torch.nn as nn


class ResidualPatchDenoiser(nn.Module):
    """
    Lightweight SimDiff-style patch Transformer denoiser for TMDM-r residual diffusion.

    Compatible with:
        V1: noise loss only
            eps = denoiser(x, y_base, r_t, t)

        V2: noise loss + r0 reconstruction loss
            eps, r0_hat = denoiser(x, y_base, r_t, t, return_r0=True)

    Inputs:
        x:       historical input series, [B, seq_len, C]
        y_base:  base forecast from NS-Transformer, [B, pred_len, C]
        r_t:     noisy residual at diffusion step t, [B, pred_len, C]
        t:       diffusion step, [B]

    Outputs:
        eps:     predicted noise, [B, pred_len, C]
        r0_hat:  predicted clean residual, [B, pred_len, C], only if return_r0=True
    """

    def __init__(self, args, num_timesteps):
        super().__init__()

        self.patch_len = getattr(args, "patch_len", 16)
        self.stride = getattr(args, "stride", 8)

        # Important:
        # Use denoiser-specific hidden size instead of args.d_model.
        # This avoids making the residual denoiser as heavy as NS-Transformer.
        self.d_model = getattr(args, "simpatch_d_model", 128)
        self.n_heads = getattr(args, "simpatch_heads", 4)
        self.n_layers = getattr(args, "simpatch_layers", 1)
        self.d_ff = getattr(args, "simpatch_d_ff", self.d_model * 2)

        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.c_out = args.c_out

        assert self.d_model % self.n_heads == 0, (
            f"simpatch_d_model ({self.d_model}) must be divisible by "
            f"simpatch_heads ({self.n_heads})."
        )

        self.time_embed = nn.Embedding(num_timesteps + 1, self.d_model)

        self.r_patch_embed = nn.Linear(self.patch_len, self.d_model)
        self.base_patch_embed = nn.Linear(self.patch_len, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=getattr(args, "dropout", 0.1),
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
        )

        self.eps_head = nn.Linear(self.d_model, self.patch_len)
        self.r0_head = nn.Linear(self.d_model, self.patch_len)

    def patchify(self, x):
        """
        x: [B, L]
        return: [B, N, patch_len]
        """
        L = x.shape[-1]

        # If L is shorter than patch_len, pad right.
        if L < self.patch_len:
            pad_len = self.patch_len - L
            x = torch.nn.functional.pad(x, (0, pad_len), mode="replicate")

        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def unpatchify(self, patches, length):
        """
        Reconstruct sequence from overlapping patches by averaging overlaps.

        patches: [B, N, patch_len]
        length:  output sequence length
        return:  [B, length]
        """
        B, N, P = patches.shape

        full_len = (N - 1) * self.stride + self.patch_len

        out = torch.zeros(B, full_len, device=patches.device, dtype=patches.dtype)
        count = torch.zeros(B, full_len, device=patches.device, dtype=patches.dtype)

        for i in range(N):
            start = i * self.stride
            end = start + self.patch_len
            out[:, start:end] += patches[:, i, :]
            count[:, start:end] += 1.0

        out = out / (count + 1e-6)

        return out[:, :length]

    def forward(self, x, y_base, r_t, t, return_r0=False):
        """
        x:       [B, seq_len, C]
        y_base:  [B, pred_len, C]
        r_t:     [B, pred_len, C]
        t:       [B]
        """
        B, L, C = r_t.shape

        # Channel-independent reshape:
        # [B, L, C] -> [B, C, L] -> [B*C, L]
        r = r_t.permute(0, 2, 1).reshape(B * C, L)
        base = y_base.permute(0, 2, 1).reshape(B * C, L)

        r_patches = self.patchify(r)
        base_patches = self.patchify(base)

        r_tokens = self.r_patch_embed(r_patches)
        base_tokens = self.base_patch_embed(base_patches)

        # t should be [B].
        # During sampling, diffusion_utils.py should expand t to [B].
        if t.dim() == 0:
            t = t.repeat(B)
        if t.numel() == 1:
            t = t.repeat(B)

        t = t.long().to(r_t.device)

        # [B] -> [B*C] -> [B*C, 1, d_model]
        t_tokens = self.time_embed(t).repeat_interleave(C, dim=0).unsqueeze(1)

        tokens = r_tokens + base_tokens + t_tokens

        hidden = self.encoder(tokens)

        eps_patches = self.eps_head(hidden)
        eps = self.unpatchify(eps_patches, L)
        eps = eps.reshape(B, C, L).permute(0, 2, 1)

        if not return_r0:
            return eps

        r0_patches = self.r0_head(hidden)
        r0_hat = self.unpatchify(r0_patches, L)
        r0_hat = r0_hat.reshape(B, C, L).permute(0, 2, 1)

        return eps, r0_hat

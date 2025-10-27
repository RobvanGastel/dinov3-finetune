import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Small utilities
# ---------------------------


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(num_layers - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvBNAct(nn.Sequential):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, act=True):
        if p is None:
            p = (k - 1) // 2
        layers = [
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


# ---------------------------
# ViT-Adapter (lite)
#   - Spatial Prior Module (SPM)
#   - Multi-scale projections (no "injector", widened dims)
# ---------------------------
class ViTAdapterLite(nn.Module):
    """
    A minimal ViT-Adapter-style front-end:
      - builds 3 multi-scale feature maps (1/8, 1/16, 1/32) from the input image features
      - no spatial-injector into the ViT (keeps your backbone frozen)
      - channel projection widened (e.g., from 1024 -> 2048) to match heavy backbones if desired

    Inputs:
      feats_in: list of tensors [B,C,H,W] from your encoder.get_intermediate_layers(..., reshape=True)
                expected order: low->high depth (we’ll internally reorder to [1/32, 1/16, 1/8])
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.p32 = ConvBNAct(in_channels, out_channels, k=1, act=True)
        self.p16 = ConvBNAct(in_channels, out_channels, k=1, act=True)
        self.p08 = ConvBNAct(in_channels, out_channels, k=1, act=True)

    def forward(self, feats_in: List[torch.Tensor]) -> List[torch.Tensor]:
        # feats_in comes as last N ViT blocks already reshaped to [B,C,H,W] by your encoder
        # We take 3 scales: deepest -> 1/32, mid -> 1/16, shallow -> 1/8
        # If your encoder gives 4 layers, we use the last three.
        assert len(feats_in) >= 3, (
            "Need at least 3 intermediate features for multi-scale."
        )
        f8 = feats_in[-3]  # highest spatial (≈1/8)
        f16 = feats_in[-2]  # (≈1/16)
        f32 = feats_in[-1]  # lowest spatial (≈1/32)

        p32 = self.p32(f32)
        p16 = self.p16(f16)
        p08 = self.p08(f8)
        return [p32, p16, p08]  # low->high resolution order


# ---------------------------
# Pixel decoder (very light)
#   - merges 1/32, 1/16, 1/8 to a robust 1/4 per-pixel embedding
# ---------------------------
class LitePixelDecoder(nn.Module):
    def __init__(self, c_in: int, c_out: int = 256):
        super().__init__()
        self.lateral32 = ConvBNAct(c_in, c_out, k=1)
        self.lateral16 = ConvBNAct(c_in, c_out, k=1)
        self.lateral08 = ConvBNAct(c_in, c_out, k=1)

        self.smooth16 = ConvBNAct(c_out, c_out, k=3)
        self.smooth08 = ConvBNAct(c_out, c_out, k=3)

        # final 1/4 embedding head
        self.embed14 = nn.Sequential(
            ConvBNAct(c_out, c_out, k=3),
            nn.Conv2d(c_out, c_out, kernel_size=1, bias=True),
        )

    def forward(
        self, feats: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # feats: [p32, p16, p08] each [B, C, H, W]
        p32, p16, p08 = feats

        l32 = self.lateral32(p32)  # 1/32
        l16 = self.lateral16(p16) + F.interpolate(
            l32, size=p16.shape[-2:], mode="nearest"
        )  # 1/16
        l16 = self.smooth16(l16)

        l08 = self.lateral08(p08) + F.interpolate(
            l16, size=p08.shape[-2:], mode="nearest"
        )  # 1/8
        l08 = self.smooth08(l08)

        # per-pixel embedding at 1/4
        emb14 = self.embed14(
            F.interpolate(l08, scale_factor=2.0, mode="bilinear", align_corners=False)
        )  # 1/4
        # return multi-scale pyramid for decoder layers to round-robin over + per-pixel embedding
        return [l32, l16, l08], emb14


# ---------------------------
# Masked-Attention Transformer Decoder (lean)
# ---------------------------
class MaskedXAttn(nn.Module):
    """Cross-attention restricted to foreground masks (binary) from previous layer."""

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        q:   [B, Nq, C]
        kv:  [B, HW, C]
        mask: binaria por query sobre K/V [B, Nq, HW] o None
        """
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)

        if mask is None:
            out, _ = self.attn(qn, kvn, kvn, key_padding_mask=None)
            return out

        outs = []
        B, Nq, HW = mask.shape
        for b in range(B):
            qb = qn[b : b + 1]  # [1, Nq, C]
            kvb = kvn[b : b + 1]  # [1, HW, C]
            outb = []
            for qi in range(Nq):
                kpm = (~mask[b, qi].bool()).unsqueeze(0)  # [1, HW]; True = “no atender”
                if kpm.all():  # <-- máscara vacía: evita NaNs
                    kpm = None  # atención global para este query
                o, _ = self.attn(qb[:, qi : qi + 1, :], kvb, kvb, key_padding_mask=kpm)
                outb.append(o)
            outb = torch.cat(outb, dim=1)  # [1, Nq, C]
            outs.append(outb)
        return torch.cat(outs, dim=0)  # [B, Nq, C]


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int = 1024):
        super().__init__()
        self.xattn = MaskedXAttn(d_model, nhead)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = MLP(d_model, dim_ff, d_model, num_layers=2)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q, kv, mask_bin):
        # (1) masked cross-attn first (as in Mask2Former)
        q = q + self.xattn(q, kv, mask_bin)
        # (2) self-attn
        qn = self.norm1(q)
        sa, _ = self.self_attn(qn, qn, qn)
        q = q + sa
        # (3) FFN
        q = q + self.ff(self.norm2(q))
        return q


class Mask2FormerHead(nn.Module):
    """
    Lean Mask2Former-style head:
      - ViT-Adapter-lite builds [1/32,1/16,1/8] features from encoder intermed. layers
      - Pixel decoder builds a 1/4 per-pixel embedding (per-pixel features)
      - Transformer decoder with masked attention refines Nq learnable queries
      - Outputs: per-query class logits and masks (H/4 x W/4) later upsampled to input size
    """

    def __init__(
        self,
        in_channels: int,  # backbone channel dim (e.g., 1024 from DINOv2)
        n_classes: int,
        n_queries: int = 100,
        d_model: int = 256,
        nheads: int = 8,
        n_layers: int = 6,
        ff_dim: int = 1024,
        upsample_masks_to: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.adapter = ViTAdapterLite(
            in_channels, out_channels=d_model
        )  # widened projection inside
        self.pixel_decoder = LitePixelDecoder(d_model, c_out=d_model)

        self.query_embed = nn.Embedding(n_queries, d_model)  # learnable query features
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, nheads, dim_ff=ff_dim) for _ in range(n_layers)]
        )

        # classification & mask heads
        self.class_head = nn.Linear(d_model, n_classes + 1)  # +1 for "no-object"
        self.mask_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, 1),
        )
        self.upsample_masks_to = upsample_masks_to

    @staticmethod
    def _hw_flatten(x: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [B,HW,C]
        return x.flatten(2).transpose(1, 2)

    def forward(
        self, inter_feats: List[torch.Tensor], img_size_hw: Tuple[int, int]
    ) -> torch.Tensor:
        """
        inter_feats: list of [B,C,H,W] tensors from encoder.get_intermediate_layers(..., reshape=True)
        img_size_hw: (H, W) of input image, for final upsampling
        Returns:
          logits_masks: [B, n_queries, n_classes, H, W] AFTER softmax over classes is left to the loss caller.
                        Commonly you’ll use class logits [B,Nq,C+1] and mask logits [B,Nq,h/4,w/4] separately.
        """
        # 1) ViT-Adapter-lite to multi-scale
        pyramid = self.adapter(inter_feats)  # [p32, p16, p08]
        # 2) Pixel decoder -> [p32,p16,p08] + per-pixel embedding @ 1/4
        feats_for_round_robin, emb14 = self.pixel_decoder(
            pyramid
        )  # list of 3 [B,C,h,w], emb14 [B,C,h4,w4]

        B, C, H4, W4 = emb14.shape
        # per_pixel = self._hw_flatten(emb14)   # [B, H4*W4, C]
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B,Nq,C]

        # initial coarse masks from per-pixel embedding vs queries (mask2former-style linear proj)
        proj = self.mask_proj(emb14)  # [B,C,H4,W4]
        mask_logits = torch.einsum("bqc, bchw -> bqhw", queries, proj)  # [B,Nq,H4,W4]
        mask_bin = (mask_logits.sigmoid() > 0.5).detach()  # binary masks for attention

        # 3) Transformer decoder with masked attention; round-robin multi-scale K/V (1/32 -> 1/16 -> 1/8 -> repeat)
        kvs = [self._hw_flatten(x) for x in feats_for_round_robin]  # [l32,l16,l08]
        sizes = [x.shape[-2:] for x in feats_for_round_robin]

        for i, layer in enumerate(self.decoder_layers):
            idx = i % len(kvs)
            kv = kvs[idx]
            h, w = sizes[idx]

            # máscara a la resolución actual
            m_res = F.interpolate(mask_bin.float(), size=(h, w), mode="nearest").to(
                torch.bool
            )  # [B,Nq,h,w]
            m_res = m_res.flatten(2)  # [B,Nq,HW]

            queries = layer(queries, kv, m_res)

            # refresca las máscaras (sigue en 1/4)
            mask_logits = torch.einsum("bqc, bchw -> bqhw", queries, proj)
            mask_bin = (mask_logits.sigmoid() > 0.5).detach()

        class_logits = self.class_head(queries)  # [B,Nq,C+1]
        masks_out = mask_logits  # [B,Nq,H4,W4]

        # upsample masks to input
        masks_up = F.interpolate(
            masks_out, size=img_size_hw, mode="bilinear", align_corners=False
        )

        return {
            "class_logits": class_logits,
            "mask_logits": masks_up,
            "mask_logits_1_4": masks_out,
        }

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRA
from .linear_decoder import LinearClassifier
from .fpn_decoder import FPNDecoder

# model/segmentation/models/__init__.py
from .segmentation.models.heads.mask2former_head import Mask2FormerHead
from .segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter


class DINOEncoderLoRA(nn.Module):
    def __init__(
        self,
        encoder,
        r: int = 3,
        emb_dim: int = 1024,
        n_classes: int = 1000,
        use_lora: bool = False,
        use_fpn: bool = False,
        use_mask2former: bool = False,
        img_dim: tuple[int, int] = (520, 520),
        m2f_hidden_dim: int = 256,
        m2f_queries: int = 100,
        m2f_dec_layers: int = 6,
        m2f_heads: int = 8,
        m2f_dim_ffn: int = 2048,
    ):
        """The DINOv2 encoder-decoder model for finetuning to downstream tasks.

        Args:
            encoder (nn.Module): The ViT encoder model loaded with the DINOv2 model weights.
            r (int, optional): The rank parameter of the LoRA weights. Defaults to 3.
            emb_dim (int, optional): The embedding dimension of the encoder. Defaults to 1024.
            n_classes (int, optional): The number of classes to output. Defaults to 1000.
            use_lora (bool, optional): Determines whether to use LoRA. Defaults to False.
            use_fpn (bool, optional): Determines whether to use the FPN decoder. Defaults to
                False.
            img_dim (tuple[int, int], optional): The input image dimension. Defaults to
                (520, 520).
        """
        super().__init__()
        assert img_dim[0] % encoder.patch_size == 0, "Wrong input shape for patches"
        assert r > 0

        self.emb_dim = emb_dim
        self.img_dim = img_dim
        self.use_lora = use_lora

        # Number of previous layers to use as input
        self.inter_layers = 4
        self.use_fpn = use_fpn
        self.use_mask2former = use_mask2former

        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Decoder
        # Patch size is given by (490/14)**2 = 35 * 35
        if self.use_fpn:
            print("Using FPN decoder")
            self.decoder = FPNDecoder(
                emb_dim,
                inter_layers=self.inter_layers,
                out_channels=128,
                patch_h=int(img_dim[0] / encoder.patch_size),
                patch_w=int(img_dim[1] / encoder.patch_size),
                n_classes=n_classes,
            )

        elif self.use_mask2former:
            print("Using Mask2Former decoder")

            # 1) Adapter con el backbone correcto + índices de ViT-L
            self.backbone = DINOv3_Adapter(
                backbone=self.encoder,  # <-- importante
                interaction_indexes=[4, 11, 17, 23],  # <-- ViT-L
                # (opcional) otros flags: with_cp=True, deform_num_heads=m2f_heads, etc.
            )
            embed_dim = self.encoder.embed_dim
            patch = self.encoder.patch_size  # 16 para ViT-L/16

            # 2) input_shape al estilo Meta (stride=4 en todas las entradas)
            input_shape = {
                "1": (embed_dim, patch * 4, patch * 4, 4),
                "2": (embed_dim, patch * 2, patch * 2, 4),
                "3": (embed_dim, patch, patch, 4),
                "4": (embed_dim, patch // 2, patch // 2, 4),
            }

            # 3) Head oficial (puedes usar hidden_dim=256 para ahorrar memoria)
            self.decoder = Mask2FormerHead(
                input_shape=input_shape,
                hidden_dim=m2f_hidden_dim,  # p. ej., 256
                num_classes=n_classes - 1,  # sin contar background
                ignore_value=255,
                transformer_in_feature="multi_scale_pixel_decoder",
            )

        else:
            print("Using linear decoder")
            self.decoder = LinearClassifier(
                emb_dim,
                patch_h=int(img_dim[0] / encoder.patch_size),
                patch_w=int(img_dim[1] / encoder.patch_size),
                n_classes=n_classes,
            )

        # Add LoRA layers to the encoder
        if self.use_lora:
            self.lora_layers = list(range(len(self.encoder.blocks)))
            self.w_a = []
            self.w_b = []

            for i, block in enumerate(self.encoder.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If the FPN decoder is used, we take the n last layers for
        # our decoder to get a better segmentation result.
        if self.use_fpn:
            # Potentially even better to take a different depths
            feature = self.encoder.get_intermediate_layers(
                x, n=self.inter_layers, reshape=True
            )
            logits = self.decoder(feature)

        elif self.use_mask2former:
            features_dict = self.backbone(x)  # dict con claves "1","2","3","4"
            out = self.decoder(
                features_dict
            )  # dict: {"pred_logits","pred_masks","aux_outputs"}
            return out

        else:
            feature = self.encoder.forward_features(x)

            # get the patch embeddings - so we exclude the CLS token
            patch_embeddings = feature["x_norm_patchtokens"]
            logits = self.decoder(patch_embeddings)

        if not self.use_mask2former:
            logits = F.interpolate(
                logits, size=x.shape[2:], mode="bilinear", align_corners=False
            )
        return logits

    def save_parameters(self, filename: str) -> None:
        """Save the LoRA weights and decoder weights to a .pt file

        Args:
            filename (str): Filename of the weights
        """
        w_a, w_b = {}, {}
        if self.use_lora:
            w_a = {f"w_a_{i:03d}": self.w_a[i].weight for i in range(len(self.w_a))}
            w_b = {f"w_b_{i:03d}": self.w_b[i].weight for i in range(len(self.w_a))}

        decoder_weights = self.decoder.state_dict()
        torch.save({**w_a, **w_b, **decoder_weights}, filename)

    def load_parameters(self, filename: str) -> None:
        """Load the LoRA and decoder weights from a file

        Args:
            filename (str): File name of the weights
        """
        state_dict = torch.load(filename)

        # Load the LoRA parameters
        if self.use_lora:
            for i, w_A_linear in enumerate(self.w_a):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_A_linear.weight = nn.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_b):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_B_linear.weight = nn.Parameter(saved_tensor)

        # Load decoder parameters
        decoder_head_dict = self.decoder.state_dict()
        decoder_head_keys = [k for k in decoder_head_dict.keys()]
        decoder_state_dict = {k: state_dict[k] for k in decoder_head_keys}

        self.decoder.load_state_dict(decoder_state_dict)

from typing import Optional
import numpy as np
from typing import Dict
import torch


def compute_iou_metric(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6,
) -> float:
    """Compute the Intersection over Union metric for the predictions and labels.

    Args:
        y_hat (torch.Tensor): The prediction of dimensions (B, C, H, W), C being
            equal to the number of classes.
        y (torch.Tensor): The label for the prediction of dimensions (B, H, W)
        ignore_index (int | None, optional): ignore label to omit predictions in
            given region.
        eps (float, optional): To smooth the division and prevent division
        by zero. Defaults to 1e-6.

    Returns:
        float: The mean IoU
    """
    num_classes = int(y.max().item() + 1)
    y_hat = torch.argmax(y_hat, dim=1)

    ious = []
    for c in range(num_classes):
        y_hat_c = y_hat == c
        y_c = y == c

        # Ignore all regions with ignore
        if ignore_index is not None:
            mask = y != ignore_index
            y_hat_c = y_hat_c & mask
            y_c = y_c & mask

        intersection = (y_hat_c & y_c).sum().float()
        union = (y_hat_c | y_c).sum().float()

        if union > 0:
            ious.append((intersection + eps) / (union + eps))

    return torch.mean(torch.stack(ious))


# =========================
#  Dice medio (mismo patrón que el IoU)
# =========================
@torch.no_grad()
def compute_dice_metric(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6,
) -> float:
    """Mean Dice across present classes (mismo criterio de presencia que IoU)."""
    num_classes = int(y.max().item() + 1)
    y_pred = torch.argmax(y_hat, dim=1)  # (B,H,W)

    dices = []
    for c in range(num_classes):
        pred_c = y_pred == c
        true_c = y == c

        if ignore_index is not None:
            mask = y != ignore_index
            pred_c = pred_c & mask
            true_c = true_c & mask

        tp = (pred_c & true_c).sum().float()
        fp = (pred_c & (~true_c)).sum().float()
        fn = ((~pred_c) & true_c).sum().float()

        denom = 2 * tp + fp + fn
        if denom > 0:
            dices.append((2 * tp + eps) / (denom + eps))

    return torch.mean(torch.stack(dices)).item() if len(dices) > 0 else float("nan")


# =========================
#  Métricas por clase: Dice e IoU
# =========================
@torch.no_grad()
def per_class_dice_iou(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6,
) -> Dict[int, Dict[str, float]]:
    """
    Devuelve un dict:
        { class_idx: {"Dice": float|nan, "IoU": float|nan} }
    """
    num_classes = int(y.max().item() + 1)
    y_pred = torch.argmax(y_hat, dim=1)  # (B,H,W)

    out: Dict[int, Dict[str, float]] = {}
    if ignore_index is not None:
        valid = y != ignore_index
    else:
        valid = torch.ones_like(y, dtype=torch.bool)

    for c in range(num_classes):
        pred_c = (y_pred == c) & valid
        true_c = (y == c) & valid

        tp = (pred_c & true_c).sum().float()
        fp = (pred_c & (~true_c)).sum().float()
        fn = ((~pred_c) & true_c).sum().float()

        # IoU
        denom_iou = tp + fp + fn
        iou = ((tp + eps) / (denom_iou + eps)).item() if denom_iou > 0 else float("nan")

        # Dice
        denom_dice = 2 * tp + fp + fn
        dice = (
            ((2 * tp + eps) / (denom_dice + eps)).item()
            if denom_dice > 0
            else float("nan")
        )

        out[c] = {"Dice": float(dice), "IoU": float(iou)}

    return out


# =========================
#  Promedio de foreground (excluye clase 0)
# =========================
@torch.no_grad()
def foreground_mean_dice_iou(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Promedio (macro) sobre clases de foreground (1..C-1) para Dice e IoU.
    Asume background = 0.
    """
    per_cls = per_class_dice_iou(y_hat, y, ignore_index, eps)
    # Tomamos todas las clases exceptuando 0 (si existe)
    keys = [k for k in per_cls.keys() if k != 0]
    if len(keys) == 0:
        return {"Dice": float("nan"), "IoU": float("nan")}

    dice_vals = torch.tensor([per_cls[k]["Dice"] for k in keys], dtype=torch.float32)
    iou_vals = torch.tensor([per_cls[k]["IoU"] for k in keys], dtype=torch.float32)

    # Ignora NaNs al promediar
    dice_mean = (
        torch.nanmean(dice_vals).item()
        if torch.any(~torch.isnan(dice_vals))
        else float("nan")
    )
    iou_mean = (
        torch.nanmean(iou_vals).item()
        if torch.any(~torch.isnan(iou_vals))
        else float("nan")
    )
    return {"Dice": float(dice_mean), "IoU": float(iou_mean)}


class DatasetSegMetrics:
    """
    Aggregates segmentation metrics over the entire dataset.
    - Macro (nnU-Net-like): per-sample, per-class Dice/IoU then nanmean over samples;
      foreground_mean excludes class 0.
    - Micro (optional): sums TP/FP/FN per class across all samples, then computes Dice/IoU.
    """

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        self.C = int(num_classes)
        self.ignore_index = ignore_index

        # Macro accumulators (per-class)
        self._dice_sum = torch.zeros(self.C, dtype=torch.float64)
        self._dice_count = torch.zeros(self.C, dtype=torch.int64)
        self._iou_sum = torch.zeros(self.C, dtype=torch.float64)
        self._iou_count = torch.zeros(self.C, dtype=torch.int64)

        # Micro accumulators (per-class)
        self._tp = torch.zeros(self.C, dtype=torch.float64)
        self._fp = torch.zeros(self.C, dtype=torch.float64)
        self._fn = torch.zeros(self.C, dtype=torch.float64)

    @torch.no_grad()
    def update(self, y_hat: torch.Tensor, y: torch.Tensor):
        pred = torch.argmax(y_hat, dim=1)

        if self.ignore_index is not None:
            valid = (y != self.ignore_index)
        else:
            valid = torch.ones_like(y, dtype=torch.bool)

        B = y.shape[0]
        for c in range(self.C):
            for b in range(B):
                pred_c = (pred[b] == c) & valid[b]
                true_c = (y[b]   == c) & valid[b]

                # get CPU float64 scalars
                tp = (pred_c & true_c).sum().to(dtype=torch.float64).cpu()
                fp = (pred_c & ~true_c).sum().to(dtype=torch.float64).cpu()
                fn = (~pred_c & true_c).sum().to(dtype=torch.float64).cpu()

                denom_iou  = tp + fp + fn
                denom_dice = (2 * tp) + fp + fn

                if denom_iou.item() > 0:
                    self._iou_sum[c]   += tp / denom_iou
                    self._iou_count[c] += 1
                if denom_dice.item() > 0:
                    self._dice_sum[c]   += (2 * tp) / denom_dice
                    self._dice_count[c] += 1

                # micro (also CPU)
                self._tp[c] += tp
                self._fp[c] += fp
                self._fn[c] += fn


    def compute(self) -> Dict[str, Dict]:
        """Returns dict with per_class, foreground_mean (macro), and micro (optional)."""
        # Macro per-class (nanmean behavior)
        per_class = {}
        fg_indices = [c for c in range(self.C) if c != 0]

        dice_per_class = torch.full((self.C,), float("nan"), dtype=torch.float64)
        iou_per_class = torch.full((self.C,), float("nan"), dtype=torch.float64)

        has_dice = self._dice_count > 0
        has_iou = self._iou_count > 0

        dice_per_class[has_dice] = self._dice_sum[has_dice] / self._dice_count[has_dice]
        iou_per_class[has_iou] = self._iou_sum[has_iou] / self._iou_count[has_iou]

        for c in range(self.C):
            per_class[c] = {
                "Dice": float(
                    dice_per_class[c].item()
                    if torch.isfinite(dice_per_class[c])
                    else float("nan")
                ),
                "IoU": float(
                    iou_per_class[c].item()
                    if torch.isfinite(iou_per_class[c])
                    else float("nan")
                ),
            }

        # Foreground macro mean (exclude 0), ignoring NaNs
        fg_dice = dice_per_class[fg_indices]
        fg_iou = iou_per_class[fg_indices]
        fg_dice_mean = (
            float(torch.nanmean(fg_dice).item())
            if torch.any(torch.isfinite(fg_dice))
            else float("nan")
        )
        fg_iou_mean = (
            float(torch.nanmean(fg_iou).item())
            if torch.any(torch.isfinite(fg_iou))
            else float("nan")
        )

        # Micro (size-weighted) per-class and foreground mean
        micro_per_class = {}
        micro_fg_vals_dice, micro_fg_vals_iou = [], []
        eps = 1e-6
        for c in range(self.C):
            tp, fp, fn = self._tp[c], self._fp[c], self._fn[c]
            dice = (
                (2 * tp + eps) / (2 * tp + fp + fn + eps)
                if (2 * tp + fp + fn) > 0
                else float("nan")
            )
            iou = (
                (tp + eps) / (tp + fp + fn + eps)
                if (tp + fp + fn) > 0
                else float("nan")
            )
            dice = float(dice)
            iou = float(iou)
            micro_per_class[c] = {"Dice": dice, "IoU": iou}
            if c != 0 and (tp + fp + fn) > 0:
                micro_fg_vals_dice.append(dice)
                micro_fg_vals_iou.append(iou)

        micro_fg = {
            "Dice": float(np.mean(micro_fg_vals_dice))
            if micro_fg_vals_dice
            else float("nan"),
            "IoU": float(np.mean(micro_fg_vals_iou))
            if micro_fg_vals_iou
            else float("nan"),
        }

        return {
            "macro": {
                "per_class": per_class,
                "foreground_mean": {"Dice": fg_dice_mean, "IoU": fg_iou_mean},
            },
            "micro": {
                "per_class": micro_per_class,
                "foreground_mean": micro_fg,
            },
        }

"""Shared rank-12 LoRA and two-head objective used in the paired runs."""

from torch import Tensor, nn
from torch.nn import functional as F


BUDGET_LORA_RANK = 12
BUDGET_LORA_ALPHA = 24
BUDGET_LORA_PREFIXES = ("backbone.detail_branch_layers.2.", "backbone.semantic_branch_layers.2.")
HEAD_LR = 0.0005
ADAPTER_LR = 0.005
DICE_WEIGHT = 0.5


class TwoHeadCE(nn.Module):
    def cross_entropies(self, logits, target):
        return tuple(F.cross_entropy(output.float(), target, ignore_index=255) for output in logits)

    def forward(self, logits, target):
        aux, main = self.cross_entropies(logits, target)
        return {"main": main, "aux": aux, "total": main + 0.4 * aux}


class TwoHeadCEDice(TwoHeadCE):
    def forward(self, logits, target):
        losses = super().forward(logits, target)
        valid = target != 255
        scores = logits[1].movedim(1, -1)[valid].float().softmax(dim=-1)
        labels = F.one_hot(target[valid], num_classes=scores.shape[-1]).to(dtype=scores.dtype)
        gt_count = labels.sum(dim=0)
        present = gt_count > 0
        overlap = (scores * labels).sum(dim=0)
        dice_by_class = (2 * overlap + 1) / (scores.sum(dim=0) + gt_count + 1)
        dice = ((1 - dice_by_class) * present).sum() / present.sum().clamp_min(1)
        return {**losses, "main_dice": dice, "total": losses["total"] + DICE_WEIGHT * dice}


class Conv2dLoRA(nn.Module):
    """Frozen convolution with a same-kernel down projection and pointwise up projection."""

    def __init__(self, base: nn.Conv2d, *, rank: int = BUDGET_LORA_RANK,
                 alpha: float = BUDGET_LORA_ALPHA) -> None:
        super().__init__()
        if base.groups != 1 or base.padding_mode != "zeros":
            raise ValueError("CamVid LoRA expects an ungrouped zero-padded convolution")
        self.base = base.requires_grad_(False)
        bound = base.weight[0].numel() ** -0.5
        down = base.weight.new_empty((rank, base.in_channels, *base.kernel_size))
        nn.init.uniform_(down, -bound, bound)
        self.down = nn.Parameter(down)
        self.up = nn.Parameter(base.weight.new_zeros((base.out_channels, rank, 1, 1)))
        self.scaling = alpha / rank

    def forward(self, image: Tensor) -> Tensor:
        hidden = F.conv2d(image, self.down, stride=self.base.stride,
                          padding=self.base.padding, dilation=self.base.dilation)
        return self.base(image) + self.scaling * F.conv2d(hidden, self.up)


def budget_lora_targets(model: nn.Module, expected_count: int = 24) -> dict[str, nn.Conv2d]:
    targets = {name: module for name, module in model.named_modules()
               if isinstance(module, nn.Conv2d) and name.startswith(BUDGET_LORA_PREFIXES)}
    if len(targets) != expected_count:
        raise ValueError(f"Expected {expected_count} convolutions in the two final GCNet stages")
    return targets


def replace_module(model: nn.Module, name: str, module: nn.Module) -> None:
    parent, attribute = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), attribute, module)


def configure_budget_lora(model: nn.Module) -> list[dict[str, float | list[nn.Parameter]]]:
    """Install the 24 adapters once and return the head and adapter SGD groups."""
    targets = budget_lora_targets(model)
    model.train().requires_grad_(True)
    model.backbone.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

    factors = []
    for name, base in targets.items():
        adapter = Conv2dLoRA(base)
        replace_module(model, name, adapter)
        factors.extend((adapter.down, adapter.up))
    return [
        {"params": list(model.decode_head.parameters()), "lr": HEAD_LR},
        {"params": factors, "lr": ADAPTER_LR},
    ]

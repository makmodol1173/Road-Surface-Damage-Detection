"""
CBAM (Convolutional Block Attention Module) — matches the definitions
used during training in yolov8s-cbam.ipynb / yolo8s-cbam-p2-1.ipynb.
This must be importable so torch can unpickle the CBAM-based checkpoints.
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, c1, ratio=16):
        super().__init__()
        hidden = max(c1 // ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


def register_cbam():
    """Register CBAM in ultralytics' namespaces + patch the model parser
    so CBAM-based checkpoints (trained in the notebooks) can be loaded."""
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.CBAM = CBAM
    modules.CBAM = CBAM

    # Patch parse_model so the CBAM layer type is recognized when a .yaml
    # architecture is parsed (not needed for loading a pickled .pt, but
    # kept for parity with the training notebooks / future retraining).
    src_file = tasks.__file__
    try:
        with open(src_file, "r", encoding="utf-8") as f:
            txt = f.read()
    except (UnicodeDecodeError, OSError):
        # Read-only install, permissions issue, or unexpected encoding —
        # this patch is optional (only needed for re-parsing a .yaml
        # architecture, not for loading an already-trained .pt), so skip
        # it quietly rather than crash the app.
        return

    if "elif m is CBAM:" not in txt:
        target = "        elif m is Concat:\n            c2 = sum(ch[x] for x in f)\n"
        replacement = (
            "        elif m is CBAM:\n"
            "            c1 = ch[f]\n"
            "            args = [c1]\n"
            "            c2 = c1\n"
            + target
        )
        if target in txt:
            txt = txt.replace(target, replacement, 1)
            try:
                with open(src_file, "w", encoding="utf-8") as f:
                    f.write(txt)
            except OSError:
                return
            import importlib
            importlib.reload(tasks)
            tasks.CBAM = CBAM

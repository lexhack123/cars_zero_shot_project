# xclip_backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import torch
from transformers import AutoModel, AutoProcessor


@dataclass
class XCLIPBackend:
    model: Any
    processor: Any
    device: str


def load_xclip(
    model_name: str = "microsoft/xclip-base-patch32",
    device: Optional[str] = None,
) -> XCLIPBackend:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return XCLIPBackend(model=model, processor=processor, device=device)
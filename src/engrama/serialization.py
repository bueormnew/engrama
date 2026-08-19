"""
ENGRAMA Model Serialization Module
Author: BUEORM
License: AGPL-3.0
"""

import os
from typing import Optional, Tuple, Union

import torch

from engrama.config import EngramaConfig
from engrama.model import EngramaModel
from engrama.tokenizer import EngramaTokenizer


def save_model(
    model: EngramaModel,
    save_dir: str,
    tokenizer: Optional[EngramaTokenizer] = None,
) -> None:
    """Save model configuration, PyTorch state_dict weights, and optional tokenizer to directory.

    Args:
        model (EngramaModel): ENGRAMA model instance.
        save_dir (str): Target directory path.
        tokenizer (Optional[EngramaTokenizer]): Tokenizer instance to save alongside model.
    """
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.json")
    model.config.save(config_path)

    weights_path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), weights_path)

    if tokenizer is not None:
        tokenizer_path = os.path.join(save_dir, "tokenizer.json")
        tokenizer.save(tokenizer_path)


def load_model(
    load_dir: str,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[EngramaModel, Optional[EngramaTokenizer]]:
    """Load model configuration, state_dict weights, and tokenizer from directory.

    Args:
        load_dir (str): Directory containing model checkpoint artifacts.
        device (Union[str, torch.device]): Target torch device.

    Returns:
        Tuple[EngramaModel, Optional[EngramaTokenizer]]: Loaded model and tokenizer (if present).
    """
    config_path = os.path.join(load_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    config = EngramaConfig.load(config_path)

    model = EngramaModel(config)
    weights_path = os.path.join(load_dir, "model.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found at: {weights_path}")

    device_target = torch.device(device) if isinstance(device, str) else device
    state_dict = torch.load(weights_path, map_location=device_target)
    model.load_state_dict(state_dict)
    model.to(device_target)

    tokenizer_path = os.path.join(load_dir, "tokenizer.json")
    tokenizer = None
    if os.path.exists(tokenizer_path):
        tokenizer = EngramaTokenizer.load(tokenizer_path)

    return model, tokenizer

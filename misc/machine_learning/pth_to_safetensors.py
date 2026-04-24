from __future__ import annotations
import torch
from safetensors.torch import save_file

def convert_pth_to_safetensors(pth_path: str, output_path: str) -> None:
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)

    # Some .pth files wrap the state dict under a key
    if not isinstance(state_dict, dict) or not all(isinstance(v, torch.Tensor) for v in state_dict.values()):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    # safetensors requires all tensors to be contiguous
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    save_file(state_dict, output_path)
    print(f"Saved to {output_path}")

convert_pth_to_safetensors("best_model.pth", "model.safetensors")
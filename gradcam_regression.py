from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import cm
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset import DEFAULT_MEAN, DEFAULT_STD
from models import build_model



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM for the age regression models.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--images", type=str, nargs="*", default=None)
    parser.add_argument("--predictions_csv", type=str, default=None)
    parser.add_argument("--select", type=str, choices=["best", "worst", "middle"], default="worst")
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()



def load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint["model_name"], **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model



def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )



def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(DEFAULT_MEAN, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(DEFAULT_STD, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)



def load_image_paths(args: argparse.Namespace) -> list[str]:
    if args.images:
        return args.images
    if args.predictions_csv:
        frame = pd.read_csv(args.predictions_csv)
        frame = frame.sort_values("abs_error", ascending=(args.select == "best"))
        if args.select == "middle":
            middle = len(frame) // 2
            half_window = max(1, args.k // 2)
            frame = frame.iloc[max(0, middle - half_window) : middle + half_window]
        else:
            frame = frame.head(args.k)
        return frame["path"].astype(str).tolist()
    raise ValueError("Provide either --images or --predictions_csv")



def compute_gradcam(model, target_layer, input_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients.append(grad_output[0].detach())

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    output = model(input_tensor)
    score = output.squeeze()
    score.backward(retain_graph=False)

    handle_fwd.remove()
    handle_bwd.remove()

    activation = activations[-1]
    gradient = gradients[-1]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam, float(score.detach().cpu().item())



def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = cm.jet(cam)[..., :3]
    overlay = (1.0 - alpha) * image + alpha * heatmap
    return np.clip(overlay, 0.0, 1.0)



def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    target_layer = model.get_gradcam_target_layer()
    transform = build_transform(args.image_size)

    image_paths = load_image_paths(args)
    for image_path in image_paths:
        path = Path(image_path)
        pil_image = Image.open(path).convert("RGB")
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        cam, prediction = compute_gradcam(model, target_layer, input_tensor)
        image_np = denormalize(input_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
        overlay = overlay_heatmap(image_np, cam)

        base_name = path.stem
        Image.fromarray((image_np * 255).astype(np.uint8)).save(output_dir / f"{base_name}_input.png")
        Image.fromarray((cam * 255).astype(np.uint8)).save(output_dir / f"{base_name}_cam.png")
        Image.fromarray((overlay * 255).astype(np.uint8)).save(
            output_dir / f"{base_name}_overlay_pred_{prediction:.2f}.png"
        )
        print(f"Saved Grad-CAM outputs for {path}")


if __name__ == "__main__":
    main()

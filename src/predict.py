"""Single-image prediction - the offline use case, and a Step 7 acceptance check.

    python -m src.predict --run runs/alexnet/20260731-120000 --image path/to/pest.jpg
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from src.config import load_config, resolve_path
from src.data.transforms import build_eval_transform, load_norm_stats
from src.evaluate import load_checkpoint
from src.utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory containing best_model.pt")
    parser.add_argument("--image", required=True, help="Path to a single image")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cpu", help="Defaults to cpu - the deployment target")
    args = parser.parse_args()

    run_dir = resolve_path(args.run)
    config = load_config(run_dir / "config.yaml", {"device": args.device})
    device = resolve_device(config["device"])

    model, checkpoint, _ = load_checkpoint(run_dir, device)
    class_names = checkpoint["class_names"]

    mean, std = load_norm_stats(resolve_path(config["norm_stats"]))
    transform = build_eval_transform(config["image_size"], mean, std)

    image_path = resolve_path(args.image)
    with Image.open(image_path) as img:
        tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]

    top_k = min(args.top_k, len(class_names))
    confidences, indices = probs.topk(top_k)

    print(f"image : {image_path.name}")
    print(f"model : {config['model']} ({run_dir.name})\n")
    for rank, (conf, idx) in enumerate(zip(confidences.tolist(), indices.tolist()), start=1):
        print(f"  {rank}. {class_names[idx]:<32} {conf * 100:6.2f}%")


if __name__ == "__main__":
    main()

# src/train.py
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from dataset import VOCDataset, get_transform
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

# ===== ARG + CONFIG =====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ===== MAIN =====
def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Config:", cfg)

    # 1️⃣ Build dataset
    dataset = VOCDataset(cfg["dataset"]["path"], cfg["dataset"]["path"].replace("images", "annotations"), transform=get_transform())
    data_loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # 2️⃣ Build model
    num_classes = cfg["dataset"]["num_classes"] + 1  # +1 for background
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=num_classes)
    model.to(device)

    # 3️⃣ Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg["training"]["learning_rate"], momentum=0.9, weight_decay=0.0005)

    # 4️⃣ Training loop
    model.train()
    for epoch in range(cfg["training"]["epochs"]):
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        for imgs, targets in data_loader:
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} loss: {losses.item():.4f}")

    # 5️⃣ Save model
    torch.save(model.state_dict(), cfg["output"]["save_path"])
    print(f"Model saved to {cfg['output']['save_path']}")


if __name__ == "__main__":
    main()
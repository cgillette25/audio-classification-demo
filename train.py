import os
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio.data import ESC50Dataset
from audio.features import LogMelSpec
from audio.model import SimpleAudioCNN
from audio.utils import set_seed


def collate_fn(batch):
    x = torch.stack([b.x for b in batch], dim=0)
    y = torch.tensor([b.y for b in batch], dtype=torch.long)
    return x, y


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))

    set_seed(cfg["train"]["seed"])
    device = torch.device(
        "cuda" if (cfg["train"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    feat = LogMelSpec(
        sample_rate=cfg["dataset"]["sample_rate"],
        n_fft=cfg["features"]["n_fft"],
        hop_length=cfg["features"]["hop_length"],
        win_length=cfg["features"]["win_length"],
        n_mels=cfg["features"]["n_mels"],
        f_min=cfg["features"]["f_min"],
        f_max=cfg["features"]["f_max"],
    )

    # Enable SpecAugment only for training
    train_ds = ESC50Dataset(
        root=cfg["dataset"]["root"],
        sample_rate=cfg["dataset"]["sample_rate"],
        clip_seconds=cfg["dataset"]["clip_seconds"],
        features=feat,
        fold=cfg["train"]["fold"],
        split="train",
        augment=True,
    )
    test_ds = ESC50Dataset(
        root=cfg["dataset"]["root"],
        sample_rate=cfg["dataset"]["sample_rate"],
        clip_seconds=cfg["dataset"]["clip_seconds"],
        features=feat,
        fold=cfg["train"]["fold"],
        split="test",
        augment=False,
    )

    model = SimpleAudioCNN(n_classes=len(train_ds.labels), dropout=cfg["model"]["dropout"]).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    loss_fn = torch.nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    history = {"train_loss": [], "test_acc": []}

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item() * x.size(0)

        train_loss = running / len(train_loader.dataset)

        # quick test accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        test_acc = correct / total

        # LR schedule step once per epoch
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = out_dir / cfg["train"]["save_name"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "labels": train_ds.labels,
                    "cfg": cfg,
                },
                save_path,
            )
            print(f"Saved best model to {save_path} (acc={best_acc:.4f})")

    # Save history for notebook plotting
    torch.save(history, out_dir / "history.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/esc50.yaml")
    args = parser.parse_args()
    main(args.config)
a
import yaml
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader

from audio.data import ESC50Dataset
from audio.features import LogMelSpec
from audio.model import SimpleAudioCNN

def collate_fn(batch):
    x = torch.stack([b.x for b in batch], dim=0)
    y = torch.tensor([b.y for b in batch], dtype=torch.long)
    return x, y

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    device = torch.device("cuda" if (cfg["train"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu")

    feat = LogMelSpec(
        sample_rate=cfg["dataset"]["sample_rate"],
        n_fft=cfg["features"]["n_fft"],
        hop_length=cfg["features"]["hop_length"],
        win_length=cfg["features"]["win_length"],
        n_mels=cfg["features"]["n_mels"],
        f_min=cfg["features"]["f_min"],
        f_max=cfg["features"]["f_max"],
    )

    test_ds = ESC50Dataset(
        root=cfg["dataset"]["root"],
        sample_rate=cfg["dataset"]["sample_rate"],
        clip_seconds=cfg["dataset"]["clip_seconds"],
        features=feat,
        fold=cfg["train"]["fold"],
        split="test",
    )
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2, collate_fn=collate_fn)

    ckpt = torch.load(f"{cfg['train']['out_dir']}/{cfg['train']['save_name']}", map_location="cpu")
    labels = ckpt["labels"]

    model = SimpleAudioCNN(n_classes=len(labels), dropout=cfg["model"]["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_true.append(y.numpy())
            y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1:  {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=labels))
    print("\nConfusion matrix shape:", cm.shape)

    # Save for notebook
    np.save(f"{cfg['train']['out_dir']}/confusion_matrix.npy", cm)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/esc50.yaml")
    args = p.parse_args()
    main(args.config)

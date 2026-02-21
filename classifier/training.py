"""
training.py — PyTorch training loop with SSE progress streaming.

Lifecycle per session:
  1. Clean up previous model/metrics files (done in views.py before this runs)
  2. Load dataset (MNIST/CIFAR auto-downloaded; custom from upload_dir)
  3. Train → push SSE events → save <session_id>.pt + <session_id>.csv
  4. Delete downloaded dataset (MNIST/CIFAR) and custom upload dir

Custom dataset format (ZIP):
  train/           <- images
  test/            <- images
  train.json       <- {"filename.jpg": "class_label", ...}
  test.json        <- same
"""
import csv
import json
import queue
import shutil
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from django.conf import settings


# ── Session registry ─────────────────────────────────────────────────────────
_sessions: dict[str, queue.Queue] = {}
_lock = threading.Lock()


def get_queue(session_id: str) -> queue.Queue:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = queue.Queue(maxsize=200)
        return _sessions[session_id]


def clear_session(session_id: str):
    with _lock:
        _sessions.pop(session_id, None)


def _push(q: queue.Queue, event: str, data: dict):
    try:
        q.put_nowait({"event": event, "data": data})
    except queue.Full:
        pass


# ── Custom dataset ────────────────────────────────────────────────────────────

class CustomImageDataset(Dataset):
    def __init__(self, base_dir: Path, split: str, transform=None):
        manifest_path = base_dir / f"{split}.json"
        img_dir = base_dir / split
        with open(manifest_path) as f:
            manifest = json.load(f)
        classes = sorted(set(manifest.values()))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(classes)
        self.samples = [
            (str(img_dir / fname), self.class_to_idx[label])
            for fname, label in manifest.items()
            if (img_dir / fname).exists()
        ]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _find_dataset_root(upload_dir: Path) -> Path:
    """Handle ZIPs that contain a single top-level folder."""
    for p in upload_dir.rglob("train.json"):
        return p.parent
    return upload_dir


def load_custom_dataset(upload_dir: str):
    base = _find_dataset_root(Path(upload_dir))
    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_ds = CustomImageDataset(base, "train", tf)
    test_ds  = CustomImageDataset(base, "test",  tf)
    return (
        DataLoader(train_ds, batch_size=32, shuffle=True),
        DataLoader(test_ds,  batch_size=64),
        train_ds.num_classes,
        (3, 64, 64),
    )


# ── Built-in datasets ─────────────────────────────────────────────────────────

DATA_DIR = str(Path(settings.BASE_DIR) / "data")


def load_dataset(name: str, upload_dir: str | None = None):
    if name == "custom":
        return load_custom_dataset(upload_dir)

    if name == "mnist":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
        test  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
        return DataLoader(train, 64, shuffle=True), DataLoader(test, 256), 10, (1, 28, 28)

    if name == "cifar10":
        tf_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=tf_train)
        test  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf_test)
        return DataLoader(train, 64, shuffle=True), DataLoader(test, 256), 10, (3, 32, 32)

    raise ValueError(f"Unknown dataset: {name}")


# ── Model builder ─────────────────────────────────────────────────────────────

ACTIVATIONS = {
    "relu":      nn.ReLU,
    "sigmoid":   nn.Sigmoid,
    "tanh":      nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
}


def build_model(layer_specs: list[dict]) -> nn.Sequential:
    layers = []
    for spec in layer_specs:
        t = spec["type"]
        if t == "conv2d":
            layers.append(nn.Conv2d(
                int(spec.get("in_channels", 1)),
                int(spec.get("out_channels", 32)),
                kernel_size=int(spec.get("kernel_size", 3)),
                padding=int(spec.get("padding", 1)),
            ))
        elif t == "maxpool2d":
            layers.append(nn.MaxPool2d(int(spec.get("kernel_size", 2))))
        elif t == "flatten":
            layers.append(nn.Flatten())
        elif t == "linear":
            layers.append(nn.Linear(int(spec["in_features"]), int(spec["out_features"])))
        elif t in ACTIVATIONS:
            layers.append(ACTIVATIONS[t]())
        elif t == "dropout":
            layers.append(nn.Dropout(float(spec.get("p", 0.5))))
        elif t == "batchnorm2d":
            layers.append(nn.BatchNorm2d(int(spec["num_features"])))
        else:
            raise ValueError(f"Unknown layer type: {t}")
    return nn.Sequential(*layers)


def build_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "adam":    return optim.Adam(params, lr=lr)
    if name == "sgd":     return optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop": return optim.RMSprop(params, lr=lr)
    if name == "adamw":   return optim.AdamW(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def build_loss(name: str):
    name = name.lower()
    if name == "crossentropy": return nn.CrossEntropyLoss()
    if name == "mse":          return nn.MSELoss()
    if name == "nllloss":      return nn.NLLLoss()
    raise ValueError(f"Unknown loss: {name}")


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(session_id: str, config: dict):
    q = get_queue(session_id)
    upload_dir = None
    model_path = None
    dataset_name = config.get("dataset", "mnist")

    try:
        epochs    = int(config.get("epochs", 5))
        lr        = float(config.get("lr", 0.001))
        opt_name  = config.get("optimizer", "adam")
        loss_name = config.get("loss", "crossentropy")
        layers    = config["layers"]

        upload_id = config.get("upload_id")
        if upload_id:
            upload_dir = str(Path(settings.MEDIA_ROOT) / "uploads" / upload_id)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _push(q, "info", {"msg": f"Using device: {device}", "device": str(device)})

        train_loader, test_loader, num_classes, _ = load_dataset(dataset_name, upload_dir)
        _push(q, "info", {"msg": f"Dataset '{dataset_name}' loaded — {num_classes} classes"})

        model     = build_model(layers).to(device)
        optimizer = build_optimizer(opt_name, model.parameters(), lr)
        criterion = build_loss(loss_name)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _push(q, "model_ready", {"total_params": total_params})

        total_batches = len(train_loader)
        epoch_metrics = []   # collected for CSV export

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = correct = total = 0

            _push(q, "epoch_start", {"epoch": epoch, "total_epochs": epochs})

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total   += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if batch_idx % 20 == 0:
                    _push(q, "batch", {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "total_batches": total_batches,
                        "loss": round(running_loss / (batch_idx + 1), 4),
                        "acc":  round(100. * correct / total, 2),
                    })

            model.eval()
            val_loss = val_correct = val_total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss    += criterion(outputs, targets).item()
                    _, predicted = outputs.max(1)
                    val_total   += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            row = {
                "epoch":      epoch,
                "train_loss": round(running_loss / total_batches, 4),
                "val_loss":   round(val_loss / len(test_loader), 4),
                "train_acc":  round(100. * correct / total, 2),
                "val_acc":    round(100. * val_correct / val_total, 2),
            }
            epoch_metrics.append(row)
            _push(q, "epoch_end", {**row, "total_epochs": epochs})

        # ── Save model (.pt) ──────────────────────────────────────────────────
        models_dir = Path(settings.MEDIA_ROOT) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{session_id}.pt"
        model.eval()
        torch.save(model.cpu(), model_path)

        # ── Save metrics (.csv) ───────────────────────────────────────────────
        metrics_dir = Path(settings.MEDIA_ROOT) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"{session_id}.csv"
        with open(metrics_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
            w.writeheader()
            w.writerows(epoch_metrics)

        _push(q, "done", {
            "msg": "Training complete!",
            "session_id": session_id,
        })

    except Exception as e:
        _push(q, "error", {"msg": str(e)})
        if model_path and model_path.exists():
            model_path.unlink(missing_ok=True)

    finally:
        # Delete custom uploaded dataset
        if upload_dir:
            shutil.rmtree(upload_dir, ignore_errors=True)

        # Delete downloaded MNIST/CIFAR data to free disk space
        if dataset_name in ("mnist", "cifar10"):
            data_dir = Path(settings.BASE_DIR) / "data"
            shutil.rmtree(data_dir, ignore_errors=True)
            _push(q, "info", {"msg": f"Dataset '{dataset_name}' files removed from disk."})


def start_training(session_id: str, config: dict):
    t = threading.Thread(target=train_model, args=(session_id, config), daemon=True)
    t.start()

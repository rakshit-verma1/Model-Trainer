"""
training.py — Universal PyTorch training loop with SSE progress streaming.

Supports:
  - ANY torch.nn module via dynamic getattr(nn, type_name) construction
  - Multiple task types: classification, regression, text/sequence
  - Multiple dataset formats: built-in (MNIST, CIFAR-10), custom image ZIP, CSV tabular
  - Special _mode handling for RNN/reshape/permute layers

Lifecycle per session:
  1. Clean up previous model/metrics files (done in views.py before this runs)
  2. Load dataset (auto-downloaded, from upload ZIP, or from uploaded CSV)
  3. Train → push SSE events → save <session_id>.pt + <session_id>.csv
  4. Delete downloaded/uploaded data
"""
import csv
import json
import math
import queue
import shutil
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import datasets, transforms

from django.conf import settings


# ── Session registry ─────────────────────────────────────────────────
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


# ── Custom Image Dataset ──────────────────────────────────────────────

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
    test_ds = CustomImageDataset(base, "test", tf)
    return (
        DataLoader(train_ds, batch_size=32, shuffle=True),
        DataLoader(test_ds, batch_size=64),
        train_ds.num_classes,
        (3, 64, 64),
    )


# ── CSV Tabular Dataset ──────────────────────────────────────────────

def load_csv_dataset(upload_dir: str, task_type: str = "classification"):
    """Load a CSV file from the upload directory. Last column = target."""
    import numpy as np

    base = Path(upload_dir)
    csv_files = list(base.rglob("*.csv"))
    if not csv_files:
        raise ValueError("No CSV file found in the uploaded archive.")

    csv_path = csv_files[0]
    data = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row:
                data.append(row)

    if not data:
        raise ValueError("CSV file is empty.")

    # Parse: all columns except last are features, last is target
    try:
        features = [[float(v) for v in row[:-1]] for row in data]
        targets_raw = [row[-1] for row in data]
    except (ValueError, IndexError) as e:
        raise ValueError(f"Error parsing CSV: {e}")

    X = torch.tensor(features, dtype=torch.float32)

    if task_type == "regression":
        # Regression: target is float
        y = torch.tensor([float(t) for t in targets_raw], dtype=torch.float32).unsqueeze(1)
        num_classes = 1
    else:
        # Classification: target is class label (string or int)
        unique_labels = sorted(set(targets_raw))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        y = torch.tensor([label_map[t] for t in targets_raw], dtype=torch.long)
        num_classes = len(unique_labels)

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    input_shape = (X.shape[1],)
    return (
        DataLoader(train_ds, batch_size=32, shuffle=True),
        DataLoader(test_ds, batch_size=64),
        num_classes,
        input_shape,
    )


# ── Built-in datasets ────────────────────────────────────────────────

DATA_DIR = str(Path(settings.BASE_DIR) / "data")


def load_dataset(name: str, upload_dir: str | None = None, task_type: str = "classification"):
    if name == "custom":
        return load_custom_dataset(upload_dir)

    if name == "csv":
        return load_csv_dataset(upload_dir, task_type)

    if name == "mnist":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
        test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
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
        train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf_train)
        test = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf_test)
        return DataLoader(train, 64, shuffle=True), DataLoader(test, 256), 10, (3, 32, 32)

    raise ValueError(f"Unknown dataset: {name}")


# ── Dynamic Model Builder ────────────────────────────────────────────

class DynamicModel(nn.Module):
    """
    Model built from a list of layer specs.
    Each spec is a dict with "type" (nn.Module class name) and constructor kwargs.
    Special "_mode" field controls forward-pass behavior for complex layers.
    """

    def __init__(self, layer_specs: list[dict]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.modes = []  # parallel list of mode strings

        for spec in layer_specs:
            spec = dict(spec)  # copy to avoid mutating original
            layer_type = spec.pop("type")
            mode = spec.pop("_mode", None)
            # Remove any private keys starting with _
            private_keys = [k for k in spec if k.startswith("_")]
            mode_data = {k: spec.pop(k) for k in private_keys}
            if mode:
                mode_data["_mode"] = mode

            # Handle Custom module type (Gemini-generated code)
            if layer_type == "Custom":
                custom_code = mode_data.get("_custom_code", "")
                custom_name = mode_data.get("_custom_name", "CustomModule")
                if not custom_code:
                    raise ValueError("Custom module requires '_custom_code'")
                # Compile and instantiate the custom module
                namespace = {"nn": nn, "torch": torch}
                try:
                    exec(custom_code, namespace)
                except Exception as e:
                    raise ValueError(f"Error compiling custom module '{custom_name}': {e}")
                if custom_name not in namespace:
                    raise ValueError(f"Custom code did not define class '{custom_name}'")
                cls = namespace[custom_name]
            else:
                # Get the nn.Module class
                if not hasattr(nn, layer_type):
                    raise ValueError(
                        f"'{layer_type}' is not a valid torch.nn module. "
                        f"Check spelling and capitalization."
                    )
                cls = getattr(nn, layer_type)

            # Convert all numeric string values to proper types
            kwargs = {}
            for k, v in spec.items():
                if isinstance(v, str):
                    try:
                        v = int(v)
                    except ValueError:
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                kwargs[k] = v

            try:
                layer = cls(**kwargs)
            except TypeError as e:
                raise ValueError(f"Error creating {layer_type}: {e}")

            self.layers.append(layer)
            self.modes.append(mode_data)

    def forward(self, x):
        # Cache outputs for skip connections
        outputs = []

        for i, (layer, mode_data) in enumerate(zip(self.layers, self.modes)):
            mode = mode_data.get("_mode")

            if mode == "rnn":
                x, _ = layer(x)
            elif mode == "select_last":
                x = layer(x)
                x = x[:, -1, :]
            elif mode == "permute":
                x = layer(x)
                dims = mode_data.get("_permute_dims", [0, 2, 1])
                x = x.permute(*dims)
            elif mode == "unsqueeze":
                x = layer(x)
                dim = int(mode_data.get("_dim", 1))
                x = x.unsqueeze(dim)
            elif mode == "squeeze":
                x = layer(x)
                dim = int(mode_data.get("_dim", 1))
                x = x.squeeze(dim)
            elif mode == "reshape":
                x = layer(x)
                shape = mode_data.get("_shape", [-1])
                x = x.reshape(x.size(0), *shape)
            elif mode == "flatten":
                x = x.flatten(1)
            else:
                x = layer(x)

            # Apply skip/residual connection: add output from a previous layer
            skip_from = mode_data.get("_skip_from")
            if skip_from is not None:
                src = int(skip_from)
                if 0 <= src < len(outputs):
                    skip_val = outputs[src]
                    if skip_val.shape == x.shape:
                        x = x + skip_val

            outputs.append(x)

        return x


def build_model(layer_specs: list[dict]) -> nn.Module:
    """Build a model from layer specs. Uses DynamicModel for full flexibility."""
    return DynamicModel(layer_specs)


def build_optimizer(name: str, params, lr: float):
    name = name.lower()
    optimizers = {
        "adam": lambda: optim.Adam(params, lr=lr),
        "sgd": lambda: optim.SGD(params, lr=lr, momentum=0.9),
        "rmsprop": lambda: optim.RMSprop(params, lr=lr),
        "adamw": lambda: optim.AdamW(params, lr=lr),
        "adagrad": lambda: optim.Adagrad(params, lr=lr),
        "adadelta": lambda: optim.Adadelta(params, lr=lr),
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    return optimizers[name]()


def build_loss(name: str):
    """Build a loss function by name. Supports any nn loss module."""
    # Try direct nn attribute first
    if hasattr(nn, name):
        return getattr(nn, name)()
    # Try common aliases
    aliases = {
        "crossentropy": "CrossEntropyLoss",
        "cross_entropy": "CrossEntropyLoss",
        "mse": "MSELoss",
        "l1": "L1Loss",
        "nllloss": "NLLLoss",
        "nll": "NLLLoss",
        "bce": "BCELoss",
        "bcewithlogits": "BCEWithLogitsLoss",
        "huber": "HuberLoss",
        "smoothl1": "SmoothL1Loss",
    }
    resolved = aliases.get(name.lower())
    if resolved and hasattr(nn, resolved):
        return getattr(nn, resolved)()
    raise ValueError(f"Unknown loss: {name}")


# ── Training loop ────────────────────────────────────────────────────

def train_model(session_id: str, config: dict):
    q = get_queue(session_id)
    upload_dir = None
    model_path = None
    dataset_name = config.get("dataset", "mnist")
    task_type = config.get("task_type", "classification")

    try:
        epochs = int(config.get("epochs", 5))
        lr = float(config.get("lr", 0.001))
        opt_name = config.get("optimizer", "adam")
        loss_name = config.get("loss", "crossentropy")
        layers = config["layers"]

        upload_id = config.get("upload_id")
        if upload_id:
            upload_dir = str(Path(settings.MEDIA_ROOT) / "uploads" / upload_id)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _push(q, "info", {"msg": f"Using device: {device}", "device": str(device)})

        train_loader, test_loader, num_classes, input_shape = load_dataset(
            dataset_name, upload_dir, task_type
        )
        _push(q, "info", {
            "msg": f"Dataset '{dataset_name}' loaded — "
                   f"{'classes: ' + str(num_classes) if task_type == 'classification' else 'regression target'}",
            "task_type": task_type,
        })

        model = build_model(layers).to(device)
        optimizer = build_optimizer(opt_name, model.parameters(), lr)
        criterion = build_loss(loss_name)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _push(q, "model_ready", {"total_params": total_params, "task_type": task_type})

        total_batches = len(train_loader)
        epoch_metrics = []

        is_regression = task_type == "regression"

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = total = 0
            mae_sum = 0.0

            _push(q, "epoch_start", {"epoch": epoch, "total_epochs": epochs})

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if is_regression:
                    mae_sum += (outputs - targets).abs().sum().item()
                    total += targets.size(0)
                else:
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                if batch_idx % 20 == 0:
                    batch_data = {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "total_batches": total_batches,
                        "loss": round(running_loss / (batch_idx + 1), 4),
                    }
                    if is_regression:
                        batch_data["mae"] = round(mae_sum / max(total, 1), 4)
                    else:
                        batch_data["acc"] = round(100.0 * correct / max(total, 1), 2)
                    _push(q, "batch", batch_data)

            # Validation
            model.eval()
            val_loss = val_correct = val_total = 0
            val_mae_sum = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    if is_regression:
                        val_mae_sum += (outputs - targets).abs().sum().item()
                        val_total += targets.size(0)
                    else:
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()

            row = {
                "epoch": epoch,
                "train_loss": round(running_loss / max(total_batches, 1), 4),
                "val_loss": round(val_loss / max(len(test_loader), 1), 4),
            }
            if is_regression:
                row["train_mae"] = round(mae_sum / max(total, 1), 4)
                row["val_mae"] = round(val_mae_sum / max(val_total, 1), 4)
            else:
                row["train_acc"] = round(100.0 * correct / max(total, 1), 2)
                row["val_acc"] = round(100.0 * val_correct / max(val_total, 1), 2)

            epoch_metrics.append(row)
            _push(q, "epoch_end", {**row, "total_epochs": epochs, "task_type": task_type})

        # ── Save model (.pt) ──────────────────────────────────────────────
        models_dir = Path(settings.MEDIA_ROOT) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{session_id}.pt"
        model.eval()
        torch.save(model.cpu(), model_path)

        # ── Save metrics (.csv) ────────────────────────────────────────────
        metrics_dir = Path(settings.MEDIA_ROOT) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"{session_id}.csv"
        fieldnames = list(epoch_metrics[0].keys()) if epoch_metrics else ["epoch"]
        with open(metrics_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
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

# ⚡ Neural Net Trainer

A browser-based image classifier trainer powered by **Django + PyTorch**. Build a neural network visually, pick a dataset and optimizer, then watch it train live — all from your browser.

![Python](https://img.shields.io/badge/Python-3.13-blue) ![Django](https://img.shields.io/badge/Django-6.0-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red)

---

## Features

- **Interactive network builder** — click layer chips to assemble a model; inline editors for every parameter (channels, features, kernel size, dropout, etc.)
- **Node-based architecture diagram** — live SVG graph of neurons and connections that updates as you build
- **Quick templates** — one-click MLP (MNIST), CNN, and Deep CNN presets
- **Built-in datasets** — MNIST and CIFAR-10 (auto-downloaded, auto-deleted after training)
- **Custom dataset** — upload a ZIP with `train/` `test/` folders and JSON label manifests
- **Live training dashboard** — epoch/batch progress, loss + accuracy charts via Server-Sent Events
- **Download outputs** — `model.pt` (full model, loadable with `torch.load`) and `metrics.csv` after training

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run dev server
uv run python manage.py runserver
```

Open **http://127.0.0.1:8000**

---

## Custom Dataset Format

Upload a ZIP with this structure:

```
dataset.zip
├── train/           ← training images
├── test/            ← validation images
├── train.json       ← {"cat1.jpg": "cat", "dog1.jpg": "dog"}
└── test.json        ← same format
```

Images are resized to 64×64. Any common format (jpg, png, webp) works.

---

## Deployment

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | insecure dev key | Django secret key — **always set in prod** |
| `DEBUG` | `True` | Set to `False` in production |
| `ALLOWED_HOSTS` | `*` | Comma-separated list e.g. `myapp.onrender.com` |

### Render / Heroku

**Build command:**
```bash
uv run python manage.py collectstatic --noinput
```

**Start command** (handled by `Procfile`):
```
gunicorn trainer.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --worker-class gthread --threads 4 --timeout 300
```

> **Why 1 worker + 4 threads?** SSE streams hold connections open for the full duration of training. Threaded mode lets uploads, SSE streams, and normal requests share the same process without starving each other.

### macOS Local Gunicorn (dev only)

macOS's Objective-C runtime conflicts with gunicorn's `fork()`. Use:

```bash
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run gunicorn trainer.wsgi:application \
  --worker-class gthread --threads 4 --workers 1 --timeout 300
```

This is **macOS-only** — not needed in production on Linux.

---

## Project Structure

```
.
├── classifier/
│   ├── training.py   ← PyTorch loop, custom dataset loader, SSE queue
│   ├── views.py      ← upload, train, SSE stream, download endpoints
│   └── urls.py
├── trainer/
│   ├── settings.py   ← env-var driven, WhiteNoise static files, no DB
│   └── wsgi.py
├── templates/
│   └── index.html    ← single-page app
├── static/
│   ├── css/style.css
│   ├── js/builder.js ← layer palette, node SVG diagram
│   └── js/train.js   ← XHR upload, SSE listener, Chart.js
└── Procfile
```

---

## How It Works

```
Browser                      Django                    PyTorch Thread
  │                            │                            │
  ├── XHR POST /upload/ ──────►│ save ZIP chunks → extract  │
  │   (progress events)        │ return upload_id            │
  │                            │                            │
  ├── POST /train/ ───────────►│ clean prev files           │
  │                            │ start_training() ─────────►│ load dataset
  │                            │                            │ build model
  ├── GET /stream/<id>/ ──────►│ StreamingHttpResponse      │ epoch loop
  │   ← SSE events             │   ← queue.get() ◄──────────│ push events
  │                            │                            │
  ├── GET /download/<id>/ ────►│ serve .pt → delete         │
  └── GET /metrics/<id>/ ─────►│ serve .csv → delete        │
```

**File lifecycle:** Dataset files are deleted after training. Model and metrics are deleted after download (or when next training starts). Nothing persists between sessions.

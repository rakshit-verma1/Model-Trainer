"""
settings.py — production-ready.

All secrets come from env vars; falls back to dev defaults.
No database — this app never needs user accounts or sessions.
WhiteNoise serves static files so no separate nginx/CDN needed.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Security ──────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "django-insecure-change-me-in-production-sza69z+ltl",
)
DEBUG = os.environ.get("DEBUG", "True") == "True"

# Accept all hosts; restrict in production via env:  HOST=myapp.com
_raw_hosts = os.environ.get("ALLOWED_HOSTS", "")
ALLOWED_HOSTS = _raw_hosts.split(",") if _raw_hosts else ["*"]

CSRF_TRUSTED_ORIGINS = [
    f"https://{h}" for h in ALLOWED_HOSTS if h not in ("*", "localhost", "127.0.0.1")
]

# ── Apps (minimal — no auth/admin needed) ────────────────────────────────────
INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "classifier",
]

# ── Middleware — WhiteNoise right after SecurityMiddleware ────────────────────
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",   # serves static files in prod
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "trainer.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "trainer.wsgi.application"

# ── No database ───────────────────────────────────────────────────────────────
# This app stores nothing persistently — training state lives in memory queues.
DATABASES = {}

# ── Internationalisation ──────────────────────────────────────────────────────
LANGUAGE_CODE = "en-us"
TIME_ZONE     = "UTC"
USE_I18N = True
USE_TZ   = True

# ── Static & media files ──────────────────────────────────────────────────────
STATIC_URL  = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"           # written by `collectstatic`
STATICFILES_DIRS = [BASE_DIR / "static"]
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

MEDIA_URL  = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# ── Upload handling ───────────────────────────────────────────────────────────
FILE_UPLOAD_MAX_MEMORY_SIZE = 2 * 1024 * 1024   # stream >2 MB to disk
DATA_UPLOAD_MAX_MEMORY_SIZE = None               # no request body size cap

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

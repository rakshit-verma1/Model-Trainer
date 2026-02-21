# Gunicorn: 1 worker with threads so SSE streams and uploads co-exist without blocking
web: gunicorn trainer.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --worker-class gthread --threads 4 --timeout 300

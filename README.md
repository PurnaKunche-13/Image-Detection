# Image Detection

Flask app for real-vs-fake face detection using the local model in `real_fake_face_classifier/`.

## Deploy securely on Render

This repo includes Render deployment files:

- `requirements.txt` installs Flask, Gunicorn, Pillow, PyTorch, Transformers, and Safetensors.
- `runtime.txt`, `.python-version`, and `render.yaml` pin Python to `3.11.11`.
- `Procfile` provides the same production Gunicorn start command as a fallback.
- `render.yaml` defines the Render web service, build command, start command, generated secret, and health check.

Recommended Render setup:

- Service type: Web Service
- Language/runtime: Python
- Build Command: `pip install --upgrade pip && pip install -r requirements.txt`
- Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180 --access-logfile - --error-logfile - --limit-request-line 4094 --limit-request-field_size 8190`
- Health Check Path: `/healthz`
- Python Version: `3.11.11`
- Environment: `FLASK_ENV=production`
- Secret: set `SECRET_KEY` in Render, or let `render.yaml` generate it for blueprint deploys.

Security notes:

- Do not run the app with Flask debug mode enabled on Render.
- Keep secrets in Render environment variables only. Do not commit `.env` files.
- Uploads are limited to 8 MB and only JPEG/PNG inputs are accepted.
- The app sends basic browser security headers including CSP, frame blocking, referrer policy, and content-type sniffing protection.

## Important model file note

`real_fake_face_classifier/model.safetensors` is larger than GitHub's normal 100 MB file limit. Use Git LFS for this file before pushing to GitHub, or host/download the model during deployment. A `.gitattributes` entry is included for `*.safetensors`, but you still need Git LFS installed and tracking the existing model file before your push.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.
"# Image-Detection" 

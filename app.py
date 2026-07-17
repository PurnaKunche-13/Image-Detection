import base64
from io import BytesIO
from pathlib import Path
import os
from flask import Flask, render_template, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_DIR = Path(__file__).resolve().parent / "real_fake_face_classifier"
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png"}
MAX_IMAGE_PIXELS = 16_000_000

try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_FILTER = Image.LANCZOS

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(32))
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_processor = None
_model = None


@app.route("/style.css")
def stylesheet():
    return send_from_directory(app.template_folder, "style.css")


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )
    return response


def load_model():
    global _processor, _model

    if _processor is not None and _model is not None:
        return _processor, _model

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")

    _processor = AutoImageProcessor.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )
    _model = AutoModelForImageClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )
    _model.eval()
    return _processor, _model


def is_allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def read_uploaded_image(uploaded_file):
    if not uploaded_file or not uploaded_file.filename:
        raise ValueError("Please upload a JPG, JPEG, or PNG image.")

    if not is_allowed_image(uploaded_file.filename):
        raise ValueError("Please upload a valid JPG, JPEG, or PNG image.")

    if uploaded_file.mimetype not in ALLOWED_IMAGE_MIME_TYPES:
        raise ValueError("Please upload a valid JPG, JPEG, or PNG image.")

    try:
        uploaded_file.stream.seek(0)
        image = Image.open(uploaded_file.stream)
        image.verify()

        uploaded_file.stream.seek(0)
        image = Image.open(uploaded_file.stream)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Please upload a valid JPG, JPEG, or PNG image.") from exc

    return image


def predict_image(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    predicted_id = int(probabilities.argmax().item())
    label = model.config.id2label.get(predicted_id, str(predicted_id))
    confidence = float(probabilities[predicted_id].item())

    return label, confidence, probabilities


def label_class(label):
    return "label-real" if label.upper() == "REAL" else "label-fake"


def preview_data_uri(image):
    display_image = image.resize((224, 224), RESAMPLE_FILTER)
    buffer = BytesIO()
    display_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def model_status():
    try:
        load_model()
    except Exception as exc:
        return False, str(exc)

    return True, None


@app.route("/healthz")
def healthz():
    return {"status": "ok"}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    model_loaded, model_error = model_status()

    if request.method == "POST":
        if not model_loaded:
            error = "Could not load the face-classification model."
        else:
            try:
                image = read_uploaded_image(request.files.get("image"))
                processor, model = load_model()
                label, confidence, probabilities = predict_image(image, processor, model)

                fake_probability = float(probabilities[0].item()) if len(probabilities) > 0 else 0.0
                real_probability = float(probabilities[1].item()) if len(probabilities) > 1 else 0.0

                result = {
                    "label": label,
                    "label_class": label_class(label),
                    "confidence": f"{confidence:.6f}",
                    "confidence_percent": f"{confidence * 100:.2f}%",
                    "fake_probability": f"{fake_probability * 100:.2f}%",
                    "real_probability": f"{real_probability * 100:.2f}%",
                    "preview_data_uri": preview_data_uri(image),
                    "width": image.width,
                    "height": image.height,
                }
            except ValueError as exc:
                error = str(exc)
            except Exception:
                app.logger.exception("Prediction failed")
                error = "Prediction failed. Please try another clear face image."

    return render_template(
        "index.html",
        error=error,
        model_dir=MODEL_DIR.name,
        model_error=model_error,
        model_loaded=model_loaded,
        result=result,
    )


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )

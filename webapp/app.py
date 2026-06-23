import base64
import os
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

from webapp.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, PORT
from webapp.disease_info import DISEASES, get_disease, list_diseases
from webapp.download_assets import DEFAULT_HF_MODEL_REPO, MODEL_PATH
from webapp.memory_utils import release_memory
from webapp.metrics_loader import get_ui_context
from webapp.model_service import get_classifier
from webapp.mosquito_facts import get_fact

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Cap preview image size returned in JSON (saves RAM on small instances)
PREVIEW_MAX_PX = int(os.environ.get("PREVIEW_MAX_PX", "640"))


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _model_file_ready() -> bool:
    return MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000


def _preview_data_url(image_bytes: bytes, mime: str) -> str:
    """Return a smaller base64 preview to limit response memory on free tier."""
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img.thumbnail((PREVIEW_MAX_PX, PREVIEW_MAX_PX), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.route("/")
def index():
    ui = get_ui_context()
    return render_template(
        "index.html",
        classes=ui["classes"],
        model_metrics=ui["model_metrics"],
        diseases=DISEASES,
    )


@app.route("/api/classes")
def api_classes():
    return jsonify({"classes": get_ui_context()["classes"]})


@app.route("/api/diseases")
def api_diseases():
    return jsonify({"diseases": list_diseases()})


@app.route("/api/diseases/<slug>")
def api_disease(slug):
    disease = get_disease(slug)
    if not disease:
        return jsonify({"error": "Disease not found."}), 404
    return jsonify({"slug": slug, **disease})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        file = request.files["image"]
        if not file.filename:
            return jsonify({"error": "Empty filename."}), 400
        if not allowed_file(file.filename):
            return jsonify({
                "error": f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            }), 400

        image_bytes = file.read()
        ground_truth = request.form.get("ground_truth", "").strip() or None

        try:
            clf = get_classifier()
        except FileNotFoundError:
            return jsonify({
                "error": "Model file not found. Redeploy with Build Command ./build.sh so the checkpoint downloads from Hugging Face.",
            }), 503
        except Exception:
            app.logger.exception("Model load failed")
            return jsonify({
                "error": (
                    "Could not load the AI model (often out of memory on Render free tier). "
                    "Try again in 30s, or upgrade to Starter plan (~$7/mo)."
                ),
            }), 503

        try:
            result = clf.predict(image_bytes)
        finally:
            release_memory()

        mime = file.mimetype or "image/jpeg"
        image_b64_url = _preview_data_url(image_bytes, mime)
        del image_bytes
        top_class = result["top_class"]
        fact = get_fact(top_class)

        response = {
            "filename": file.filename,
            "image_data_url": image_b64_url,
            "prediction": result,
            "fact": fact,
            "model_metrics": get_ui_context()["model_metrics"],
        }

        if ground_truth:
            if ground_truth not in clf.classes:
                response["ground_truth_evaluation"] = {
                    "ground_truth": ground_truth,
                    "error": "Ground truth label is not in the trained class list.",
                }
            else:
                response["ground_truth_evaluation"] = clf.evaluate_ground_truth(
                    top_class, ground_truth
                )

        return jsonify(response)

    except Exception:
        app.logger.exception("Prediction request failed")
        return jsonify({
            "error": "Prediction failed on the server. Please try again in a moment.",
        }), 500


@app.route("/health")
def health():
    """Lightweight health check — does not load PyTorch (avoids Render worker timeout)."""
    return jsonify({
        "status": "ok",
        "model_file_ready": _model_file_ready(),
        "model_path": str(MODEL_PATH),
        "hf_model_repo": os.environ.get("HF_MODEL_REPO", DEFAULT_HF_MODEL_REPO),
    })


if __name__ == "__main__":
    print("Loading model...")
    get_classifier()
    print(f"Starting MosquitoScope on http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)

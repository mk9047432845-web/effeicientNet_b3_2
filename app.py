from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, requests, gc
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

# =====================
# CONFIG & PATHS
# =====================
app = Flask(__name__)
CORS(app)

MODEL_DIR = "models_cache"
UPLOAD_FOLDER = "uploads"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_LABELS = ["benign", "malignant", "normal"]
ALLOWED_EXT = {"jpg", "jpeg", "png"}

# =====================
# MODEL CONFIG (ONLY B3)
# =====================
MODEL_URL = "https://huggingface.co/mani880740255/skin_care_tflite/resolve/main/efficientnet_b3_skin_cancer.pth"
MODEL_PATH = os.path.join(MODEL_DIR, "efficientnet_b3.pth")

# =====================
# CHAT DATA
# =====================
CHAT_RESPONSES = {
    "what is skin care?": "Skin care is the practice of maintaining healthy, clean, and protected skin.",
    "what is a benign lesion?": "A benign lesion is non-cancerous and does not spread.",
    "what is a malignant lesion?": "A malignant lesion is cancerous and can spread.",
    "signs of skin cancer": "Irregular shape, color change, bleeding, rapid growth.",
    "how to prevent skin cancer?": "Use sunscreen, avoid excess sun, wear protective clothing."
}

# =====================
# HELPERS
# =====================
def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print("Downloading EfficientNet-B3 model...")
        r = requests.get(MODEL_URL, stream=True)
        if r.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        else:
            raise Exception("Model download failed")

# =====================
# EFFICIENTNET-B3 PREDICT
# =====================
def predict_b3(img_path):
    ensure_model_exists()

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = torch.nn.Linear(1536, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0].tolist()

    del model
    gc.collect()

    idx = int(np.argmax(probs))
    return idx, probs

# =====================
# ROUTES
# =====================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image required"}), 400

    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    try:
        idx, probs = predict_b3(path)
        return jsonify({
            "model_used": "efficientnet_b3",
            "prediction": CLASS_LABELS[idx],
            "confidence": float(probs[idx]),
            "probabilities": {
                CLASS_LABELS[i]: probs[i] for i in range(3)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)

# Example Flask Backend
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get("message", "").lower().strip() # Clean the input

    # 1. Provide initial suggestions if message is empty
    if user_msg == "":
        return jsonify({
            "suggestions": list(CHAT_RESPONSES.keys())[:3] # Returns first 3 keys as buttons
        })
    
    # 2. Check if the user's question exists in our dictionary
    if user_msg in CHAT_RESPONSES:
        return jsonify({
            "reply": CHAT_RESPONSES[user_msg],
            "suggestions": []
        })
    
    # 3. Fallback if question isn't recognized
    return jsonify({
        "reply": "I'm sorry, I only answer specific skin health questions. Try using the suggested buttons.",
        "suggestions": list(CHAT_RESPONSES.keys())
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

import os
import json
import uuid
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# =========================================================
# BASE PATHS (VERY IMPORTANT FOR YOUR STRUCTURE)
# =========================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(STATIC_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================================================
# FLASK APP
# =========================================================
app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)

# =========================================================
# LOAD MODEL & CLASSES
# =========================================================
import os
import json
import tensorflow as tf

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(BASE_DIR, "best_eefficientnet_model")
CLASS_PATH = os.path.join(BASE_DIR, "class_indices.json")

print("BASE_DIR =", BASE_DIR)
print("MODEL_PATH =", MODEL_PATH)
print("CLASS_PATH =", CLASS_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

# =========================================================
# FIND LAST CONV LAYER (FOR GRAD-CAM)
# =========================================================
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            return layer.name

last_conv_layer = get_last_conv_layer(model)

# =========================================================
# GRAD-CAM FUNCTION
# =========================================================
def make_gradcam(img_array):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy(), predictions[0].numpy(), class_idx.numpy()

# =========================================================
# ROUTES
# =========================================================
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        file = request.files["image"]

        # -------------------------------
        # SAFE UNIQUE FILENAME
        # -------------------------------
        ext = os.path.splitext(file.filename)[1]
        filename = secure_filename(f"{uuid.uuid4().hex}{ext}")

        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)

        # -------------------------------
        # PREPROCESS IMAGE
        # -------------------------------
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # -------------------------------
        # PREDICTION + GRAD-CAM
        # -------------------------------
        heatmap, preds, class_id = make_gradcam(img_array)

        label = class_names[class_id]
        confidence = preds[class_id] * 100

        # -------------------------------
        # CREATE GRAD-CAM IMAGE
        # -------------------------------
        original = cv2.imread(img_path)
        original = cv2.resize(original, (224, 224))

        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        gradcam = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        output_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, gradcam)

        # -------------------------------
        # DEBUG CHECK (OPTIONAL)
        # -------------------------------
        print("UPLOADS:", os.listdir(UPLOAD_FOLDER))
        print("OUTPUTS:", os.listdir(OUTPUT_FOLDER))

        return render_template(
            "result.html",
            label=label,
            confidence=f"{confidence:.2f}",
            original=filename,
            gradcam=filename
        )

    return render_template("index.html")

# =========================================================
# RUN APP
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)

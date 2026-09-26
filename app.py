from flask import Flask, request, jsonify
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

import numpy as np
import os


app = Flask(__name__)


MODEL_PATH = "fruit_model.tflite"


class_names = [
    "apple",
    "banana",
    "cherry",
    "chickoo",
    "grapes",
    "kiwi",
    "mango",
    "orange",
    "strawberry",
]


# Load model once when server starts
print("Loading LiteRT model...")

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("LiteRT model loaded successfully!")
print("Input shape:", input_details[0]["shape"])
print("Input dtype:", input_details[0]["dtype"])
print("Output shape:", output_details[0]["shape"])
print("Output dtype:", output_details[0]["dtype"])


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "message": "Fruit AI API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "no image found"
        }), 400

    try:
        image_file = request.files["image"]

        # Open image
        image = Image.open(image_file).convert("RGB")

        # Resize to model input size
        image = image.resize((224, 224))

        # Convert image to float32
        image_array = np.array(
            image,
            dtype=np.float32
        )

        # Add batch dimension
        image_batch = np.expand_dims(
            image_array,
            axis=0
        )

        print("Received image. Starting prediction...")

        # Send image to LiteRT
        interpreter.set_tensor(
            input_details[0]["index"],
            image_batch
        )

        # Run inference
        interpreter.invoke()

        # Get predictions
        predictions = interpreter.get_tensor(
            output_details[0]["index"]
        )

        # Find highest prediction
        predicted_index = int(
            np.argmax(predictions[0])
        )

        confidence = float(
            predictions[0][predicted_index]
        )

        prediction = class_names[predicted_index]

        print("Prediction completed.")
        print("Fruit:", prediction)
        print("Confidence:", confidence)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

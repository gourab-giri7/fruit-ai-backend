from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app=Flask(__name__)
model = tf.keras.models.load_model(
    'fruit_model.keras',
    custom_objects={
        'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input
    }
)
print("Fruit model loaded successfully!")
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
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "no image found",
        }),400
    
    image_file = request.files['image']

    image=Image.open(image_file).convert("RGB")
    image=image.resize((224,224))

    image_array = np.array(image)
    image_batch = np.expand_dims(image_array, axis=0)

    print("Received image. Starting prediction...")

    predictions = model.predict(
    image_batch,
    verbose=0
    )

    print("Prediction completed.")

    predicted_index = int(np.argmax(predictions[0]))

    confidence = float(predictions[0][predicted_index])

    return jsonify({
        "prediction": class_names[predicted_index],
        "confidence": confidence
    })
if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )
    app.run(host="0.0.0.0", port=5000, debug=True)
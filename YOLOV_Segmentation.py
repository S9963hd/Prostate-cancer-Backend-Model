from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all requests

# Apply CORS headers manually (for safety)
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# Initialize Roboflow API
RF_API_KEY = "8Qa7KjefjGCaCBRyRn7D"
rf = Roboflow(api_key=RF_API_KEY)
project = rf.workspace().project("tissue-v2")
model = project.version(4).model

def process_image(image_path):
    """Process the image using Roboflow and Supervision."""
    print(f"Processing image: {image_path}")

    if not os.path.exists(image_path):
        print("Error: Image file not found!")
        return None

    image = cv2.imread(image_path)

    try:
        results = model.predict(image_path, confidence=40, overlap=30).json()
    except Exception as e:
        print("Error during model prediction:", e)
        return None

    xyxy, confidences, class_ids = [], [], []
    class_name_to_id = {}

    for prediction in results.get('predictions', []):
        x1 = prediction['x'] - prediction['width'] / 2
        y1 = prediction['y'] - prediction['height'] / 2
        x2 = prediction['x'] + prediction['width'] / 2
        y2 = prediction['y'] + prediction['height'] / 2
        xyxy.append([x1, y1, x2, y2])
        confidences.append(prediction['confidence'])

        class_name = prediction['class']
        if class_name not in class_name_to_id:
            class_name_to_id[class_name] = len(class_name_to_id)
        class_ids.append(class_name_to_id[class_name])

    detections = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int)
    )

    bounding_box_annotator = sv.BoxAnnotator()
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)

    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy.astype(int)):
        class_id = detections.class_id[i]
        label = list(class_name_to_id.keys())[class_id]
        confidence = detections.confidence[i]
        cv2.putText(
            annotated_image,
            f"{label} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    output_path = "./annotated_image.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"Annotated image saved: {output_path}")

    return output_path

@app.route("/process", methods=["POST"])
def process():
    """API Endpoint to process the uploaded image."""
    if "file" not in request.files:
        print("Error: No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_path = "./uploaded_image.jpg"
    file.save(image_path)

    print(f"Received file: {file.filename}, saved to {image_path}")

    output_path = process_image(image_path)
    if not output_path:
        return jsonify({"error": "Failed to process image"}), 500

    print("Sending processed image response")
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get port from Render environment
    print(f"Starting Flask server on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
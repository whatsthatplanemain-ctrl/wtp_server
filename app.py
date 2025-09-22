from flask import Flask, request, jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

# Load the TorchScript model
# Make sure the filename matches exactly what you pushed to GitHub
MODEL_PATH = "model_aircraft.pt"
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

# Define preprocessing (adjust if your training pipeline was different)
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # resize to model’s expected input
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # standard ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

# Example labels (replace with your actual aircraft classes)
labels = [
    "Airbus A320",
    "Boeing 737",
    "Boeing 747",
    "Boeing 777",
    "Concorde",
    "F-16 Fighting Falcon",
    "Cessna 172"
]

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Aircraft recognition API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(file.stream).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            class_id = predicted.item()
            class_name = labels[class_id] if class_id < len(labels) else str(class_id)

        return jsonify({"class_id": class_id, "class_name": class_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local testing only; Render will use gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)

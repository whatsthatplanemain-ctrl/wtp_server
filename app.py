from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

app = Flask(__name__)

# Load your trained model (safe because it's your own file)
model = torch.load("model_aircraft.pt", map_location="cpu", weights_only=False)
model.eval()

# Define preprocessing (adjust to match your training pipeline)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Aircraft classes (alphabetical order of your dataset folders)
labels = [
    "737",
    "747",
    "757",
    "777",
    "a310",
    "a320",
    "a330",
    "a330-beluga",
    "a340",
    "a350",
    "an_124",
    "cessna172",
    "eurofighter_typhoon",
    "not_planes"
]

@app.route("/")
def home():
    return "🚀 Aircraft recognition server is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = outputs.max(1)
        result = labels[predicted.item()]

    return jsonify({"result": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port)

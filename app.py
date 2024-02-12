from flask import Flask, render_template, request, jsonify
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from werkzeug.utils import secure_filename
import os

# Initialize Flask app
app = Flask(__name__)

# Define transforms to apply to the images
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),         # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Load the model
classes = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'cotton_bacterial_blight', 'cotton_curl_virus', 'cotton_fussarium_wilt', 'cotton_healthy']  # Replace with the actual class names from your dataset

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(57600, 128)  # Adjusted input size to match flattened tensor size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Load the model
model_path = 'plant_disease_model.pth'
model = CNN(num_classes=len(classes))
model.load_state_dict(torch.load(model_path))
model.eval()

# Define function to predict disease and get advisory information
def predict_disease(image):
    image_tensor = transform(image).unsqueeze(0)
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)[0] * 100
    confidence, predicted_class_idx = torch.max(probabilities, dim=0)
    disease = classes[predicted_class_idx]
    advisory_info = get_advisory_info(disease)
    return disease, confidence.item(), advisory_info

def get_advisory_info(disease):
    if disease == "Potato___Early_blight":
        return "Prevent early blight by rotating crops and using fungicides."
    elif disease == "Potato___Late_blight":
        return "Prevent late blight by using disease-resistant potato varieties and practicing good crop management."
    elif disease == "Potato___healthy":
        return "The potato plant appears healthy. Continue to monitor for any signs of disease and maintain good plant care practices."
    elif disease == "cotton_bacterial_blight":
        return "not yet written"
    elif disease == "cotton_curl_virus":
        return "not yet written"
    elif disease == "cotton_fussarium_wilt":
        return "not yet written"
    elif disease == "cotton_healthy":
        return "not yet written"
    else:
        return "No specific advisory information available for this disease."

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        # If file is selected, process it and predict the disease
        if file:
            image = Image.open(file)
            disease, confidence, advisory_info = predict_disease(image)
            return jsonify({'disease': disease, 'confidence': confidence, 'advisory_info': advisory_info})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
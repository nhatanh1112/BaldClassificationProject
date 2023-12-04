import torch
import torchvision
import io
import streamlit as st
from PIL import Image
from facedetector import FaceDetector

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
mobilenet_v3_model = torchvision.models.mobilenet_v3_small(num_classes=2)
mobilenet_v3_model.load_state_dict(torch.load("classificationMobileNetV3v2.pth", map_location=torch.device('cpu')))
mobilenet_v3_model.eval()

regnet_y_model = torchvision.models.regnet_y_400mf(num_classes=2)
regnet_y_model.load_state_dict(torch.load("classificationRegNetY400mf.pth", map_location=torch.device('cpu')))
regnet_y_model.eval()

shufflenet_v2_model = torchvision.models.shufflenet_v2_x1_0(num_classes=2)
shufflenet_v2_model.load_state_dict(torch.load("classificationShuffleNetv2x1_0.pth", map_location=torch.device('cpu')))
shufflenet_v2_model.eval()

# Create an instance of FaceDetector
face_detector = FaceDetector()

def predict_baldness(uploaded_file, model):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((96, 96)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Open the uploaded file as an image
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

    # Preprocess the image using FaceDetector class
    faces_data = face_detector.detect_face(image, resized=True, size=96, margin=(20, 20, 40, 10))

    if not faces_data:
        st.write("No face detected in the image.")
        return

    predictions = []

    # Perform prediction for each detected face
    for face_data in faces_data:
        face_image = transform(face_data["image"]).unsqueeze(0)

        # Perform prediction
        with torch.no_grad():
            output = model(face_image)

        # Get the predicted probabilities for each class
        probabilities = torch.softmax(output, dim=1).squeeze().tolist()
        predictions.append(probabilities[1])

        processed_images = [face_data["image"] for face_data in faces_data]

    return processed_images, predictions

# Streamlit UI
st.title("Bald Classification")
st.write("Upload an image to predict whether the person in the image is bald or not.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Model selection dropdown
selected_model = st.selectbox("Select Model", ["MobileNetV3", "ShuffleNetV2", "RegNetY"])

# Make prediction and display result
if uploaded_file:
    if selected_model == "MobileNetV3":
        model = mobilenet_v3_model
    elif selected_model == "RegNetY":
        model = regnet_y_model
    elif selected_model == "ShuffleNetV2":
        model = shufflenet_v2_model
    
    processed_images, probabilities = predict_baldness(uploaded_file, model)
    if processed_images and probabilities:
        for idx, (image, prediction) in enumerate(zip(processed_images, probabilities)):
            st.write(f"THE PREDICTION RESULTS")
            st.image(uploaded_file, caption='Upload Image', width=200)
            st.image(image, caption='Detected Faces', width=200)
            face_number = idx + 1
            bald_probability = prediction
            not_bald_probability = 1 - prediction
            prediction_text = f"Prediction for Face {face_number}: {bald_probability:.2f} Bald, {not_bald_probability:.2f} Not Bald"
            st.write(prediction_text)
            if bald_probability > 0.5:
                st.write(f"=> Bald")
            else:
                st.write(f"=> Not Bald")

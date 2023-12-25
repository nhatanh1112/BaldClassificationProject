import torch
import torchvision
import io
import streamlit as st
from PIL import Image
from facedetector import FaceDetector

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained models
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

    # Create an empty placeholder for the warning message
    warning_placeholder = st.empty()

    if not faces_data:
        # No faces detected, update the placeholder with the warning message
        warning_placeholder.warning("No face detected in the uploaded image.")
        return None, None

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
st.write("Upload images to predict whether the persons in the images are bald or not.")

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Model selection dropdown
selected_model = st.selectbox("Select Model", ["MobileNetV3", "ShuffleNetV2", "RegNetY"])

# Make predictions and display results for each image
if uploaded_files:
    if selected_model == "MobileNetV3":
        model = mobilenet_v3_model
    elif selected_model == "RegNetY":
        model = regnet_y_model
    elif selected_model == "ShuffleNetV2":
        model = shufflenet_v2_model
    
    for idx, uploaded_file in enumerate(uploaded_files):
        processed_images, probabilities = predict_baldness(uploaded_file, model)
        if processed_images and probabilities:
            for face_number, (image, prediction) in enumerate(zip(processed_images, probabilities)):
                st.write(f"THE PREDICTION RESULTS for Image {idx + 1}, Face {face_number + 1}")
                st.image(uploaded_file, caption='Upload Image', width=200)
                st.image(image, caption='Detected Faces', width=200)
                bald_probability = prediction
                not_bald_probability = 1 - prediction
                prediction_text = f"Prediction: {bald_probability:.2f} Bald, {not_bald_probability:.2f} Not Bald"
                st.write(prediction_text)
                if bald_probability > 0.5:
                    st.write(f"=> Bald")
                else:
                    st.write(f"=> Not Bald")

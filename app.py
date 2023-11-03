import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Load the trained model
model = torchvision.models.resnet34(num_classes=2)
model.load_state_dict(torch.load("classification.pth"))
model.eval()

# Define the prediction function
def predict_baldness(image):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Preprocess the image
    image = Image.open(image).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)

    # Get the predicted class (0 for NotBald, 1 for Bald)
    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

# Streamlit UI
st.title("Bald Classification")
st.write("Upload an image to predict whether the person in the image is bald or not.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Make prediction and display result
if uploaded_file:
    prediction = predict_baldness(uploaded_file)
    if prediction == 1:
        result = "Bald"
    else:
        result = "Not Bald"

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"Prediction: {result}")

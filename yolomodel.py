import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import os

# Load the trained YOLO model
model = YOLO("best1.pt")  # Make sure best.pt is in the same folder

st.title("YOLOv8 Object Detection App")
st.write("Upload an image to detect and count objects using a trained YOLOv8 model.")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Perform detection
    results = model(image)

    # Get bounding box class IDs
    class_ids = results[0].boxes.cls.tolist()
    class_names = model.names

    # Count the number of each detected class
    counts = Counter([class_names[int(cls)] for cls in class_ids])

    # Show counts
    st.subheader("🔢 Detected Object Counts")
    if counts:
        for cls, count in counts.items():
            st.write(f"- **{cls}**: {count}")
    else:
        st.write("No objects detected.")

    # Save and show result image with bounding boxes
    annotated_img_path = "output.jpg"
    results[0].save(filename=annotated_img_path)

    st.subheader("📸 Image with Bounding Boxes")
    st.image(annotated_img_path, use_column_width=True)

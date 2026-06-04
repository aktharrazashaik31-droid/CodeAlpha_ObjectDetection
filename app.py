import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page Settings
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🎯"
)

st.title("🎯 Object Detection using YOLOv8")

# Load Model
model = YOLO("yolov8n.pt")

# Upload Image
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read Image
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Convert to numpy array
    img_array = np.array(image)

    # Detection
    results = model(img_array, conf=0.25)

    # Draw boxes
    annotated_image = results[0].plot()

    st.subheader("Detection Result")
    st.image(annotated_image, use_container_width=True)

    # Extract Objects
    detected_objects = []

    if len(results[0].boxes) > 0:

        st.subheader("Detected Objects")

        for box in results[0].boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            object_name = model.names[class_id]

            detected_objects.append(object_name)

            st.write(
                f"✅ {object_name} ({confidence:.2f})"
            )

        st.success(
            f"Total Objects Detected: {len(detected_objects)}"
        )

    else:
        st.warning("No objects detected.")
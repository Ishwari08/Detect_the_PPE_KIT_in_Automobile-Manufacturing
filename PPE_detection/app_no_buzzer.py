import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import os

# Load the YOLO model
model = YOLO("D:/C DRIVE/Downloads/PPE_Detection/ppe_detection/best1.pt")

# Define class colors
classColors = {
    'Person': (255, 0, 0),           # Blue
    'Hardhat': (238, 130, 238),      # Violet
    'Safety Vest': (0, 255, 0),      # Green
    'Mask': (139, 69, 19),           # Brown
    'Gloves': (0, 255, 255),         # Yellow
    'NO-Hardhat': (0, 0, 255),       # Red
    'NO-Mask': (0, 0, 255),          # Red
    'NO-Safety Vest': (0, 0, 255),   # Red
    'NO-Person': (0, 0, 255),        # Red
    'NO-Gloves': (0, 0, 255)         # Red
}

# Define Streamlit App
st.title("YOLO Object Detection App")
st.write("Upload a video, and the model will perform object detection.")

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mkv"])

if uploaded_file is not None:
    # Save uploaded file to a temporary directory
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.write("### Original Video:")
    st.video(tfile.name)

    # Process video
    st.write("### Processing Video...")

    # OpenCV processing
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Temporary output file
    output_path = "processed_video.mp4"  # Use a fixed path for debugging
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    processed_frames = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform YOLO detection
        results = model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]) * 100, 2)
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # Get color for each class
                color = classColors.get(class_name, (255, 255, 255))  # Default to white if class not found
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} {conf}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Log detected objects
                print(f"Detected {class_name} with confidence {conf}% at ({x1}, {y1}, {x2}, {y2})")

        # Write processed frame
        if frame is not None:
            out.write(frame)
        else:
            print("Empty frame encountered!")

        # Update progress bar
        processed_frames += 1
        progress_bar.progress(processed_frames / frame_count)

    # Release resources
    cap.release()
    out.release()

    # Display processed video
    st.write("### Processed Video:")
    st.video(output_path)

    # Download button
    with open(output_path, "rb") as file:
        st.download_button("Download Processed Video", file, "processed_video.mp4")

import cv2
import numpy as np
import streamlit as st
import time
from ultralytics import YOLO
from io import BytesIO
import tempfile

# Load YOLOv8 model
model_path = r"C:\Users\Kaushik\code_JN\runs\detect\train2\weights\best.pt"
model = YOLO(model_path)

# Streamlit UI
st.title("YOLOv8 Streamlit App")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ("Live Webcam Feed", "Upload Video"))

# Heatmap variables
heatmap_accumulated = None
decay_factor = 0.997  # Adjusts fading speed of the heatmap trail

# Initialize variables to store detections and last update time
last_detection_time = time.time()  # To control when to update the detection display
current_detection_count = 0  # This will hold the displayed detection count
last_displayed_detection_count = 0  # This will hold the detection count to be displayed


# ---------------------------- Live Webcam Feed ---------------------------- #

if page == "Live Webcam Feed":
    # Start/Stop control
    if "run_stream" not in st.session_state:
        st.session_state.run_stream = False

    def toggle_stream():
        st.session_state.run_stream = not st.session_state.run_stream

    st.button("Start/Stop Stream", on_click=toggle_stream, key="toggle_button")

    # Image placeholder
    FRAME_WINDOW = st.image([])

    # Stream video if enabled
    if st.session_state.run_stream:
        cap = cv2.VideoCapture(0)  # Default webcam

        while st.session_state.run_stream:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture frame.")
                break

            # Run YOLOv8 detection
            results = model(frame)

            # Get bounding box centers
            centers = []
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = box[:4]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                centers.append((center_x, center_y))

            # Initialize heatmap on first frame
            if heatmap_accumulated is None:
                heatmap_accumulated = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

            # Apply decay to fade old trails
            heatmap_accumulated *= decay_factor

            # Add new detections to the heatmap
            for center in centers:
                cv2.circle(heatmap_accumulated, center, 25, (255), thickness=-1)

            # Smooth the heatmap
            heatmap_blurred = cv2.GaussianBlur(heatmap_accumulated, (101, 101), 0)

            # Normalize and apply colormap
            heatmap_normalized = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_color = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

            # Overlay heatmap on the frame
            overlay = cv2.addWeighted(frame, 0.7, heatmap_color, 0.5, 0)

            # Update the detection count only after 1 second delay
            current_detection_count = len(results[0].boxes)

            # Only update the displayed count after 1 second has passed
            if time.time() - last_detection_time >= 1:
                last_displayed_detection_count = current_detection_count
                last_detection_time = time.time()

            # Display the detection count with 1-second delay
            cv2.putText(
                overlay, f"Detections: {last_displayed_detection_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # Convert to RGB and update Streamlit image
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(overlay_rgb, channels="RGB")

        cap.release()
        st.write("Webcam stopped.")

# ---------------------------- Video Upload for Processing ---------------------------- #

if page == "Upload Video":
    # File uploader for video
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        # Save uploaded video to a temporary file
        video_bytes = uploaded_video.read()
        with tempfile.NamedTemporaryFile(delete=False) as tmp_video:
            tmp_video.write(video_bytes)
            tmp_video_path = tmp_video.name

        # Decode the video file and stream it
        cap = cv2.VideoCapture(tmp_video_path)

        # Streamlit image placeholder for video frames
        FRAME_WINDOW = st.image([])

        # Initialize heatmap for video processing
        heatmap_accumulated = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv8 detection on video frames
            results = model(frame)

            # Get bounding box centers
            centers = []
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = box[:4]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                centers.append((center_x, center_y))

            # Initialize heatmap on first frame
            if heatmap_accumulated is None:
                heatmap_accumulated = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

            # Apply decay to fade old trails
            heatmap_accumulated *= decay_factor

            # Add new detections to the heatmap
            for center in centers:
                cv2.circle(heatmap_accumulated, center, 25, (255), thickness=-1)

            # Smooth the heatmap
            heatmap_blurred = cv2.GaussianBlur(heatmap_accumulated, (101, 101), 0)

            # Normalize and apply colormap
            heatmap_normalized = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_color = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

            # Overlay heatmap on the frame
            overlay = cv2.addWeighted(frame, 0.7, heatmap_color, 0.5, 0)

            # Update the detection count only after 1 second delay
            current_detection_count = len(results[0].boxes)

            # Only update the displayed count after 1 second has passed
            if time.time() - last_detection_time >= 0.5:
                last_displayed_detection_count = current_detection_count
                last_detection_time = time.time()

            # Display the detection count with 1-second delay
            cv2.putText(
                overlay, f"Detections: {last_displayed_detection_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # Display the frame with heatmap and detection count
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(overlay_rgb, channels="RGB")

        cap.release()
        st.write("Video processing complete.")

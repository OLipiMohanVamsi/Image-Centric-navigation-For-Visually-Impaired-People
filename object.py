import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pyttsx3
import tempfile

# Set Streamlit page configuration first to avoid StreamlitSetPageConfigMustBeFirstCommandError
st.set_page_config(page_title="Image-Centric Indoor Navigation Solution for Visually Impaired People", page_icon="🖼️")

# Load YOLOv8 model using ultralytics
@st.cache_resource
def load_model():
    return YOLO('yolov8s.pt')  # Replace 'yolov8s.pt' with other variants if needed

model = load_model()

def detect_objects(image):
    """Detect objects in an image using YOLOv8."""
    results = model(image)  # Using YOLOv8's detection method
    return results

def analyze_position(bbox, image_width, image_height):
    """Analyze the position of the object in the image."""
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Determine horizontal position
    if x_center < image_width / 3:
        horizontal = "Left"
    elif x_center > 2 * image_width / 3:
        horizontal = "Right"
    else:
        horizontal = "Center"

    # Determine vertical position
    if y_center < image_height / 3:
        vertical = "Top"
    elif y_center > 2 * image_height / 3:
        vertical = "Bottom"
    else:
        vertical = "Center"

    # Determine corner positions (Optional)
    if x_center < image_width / 3 and y_center < image_height / 3:
        corner = "Top-Left"
    elif x_center > 2 * image_width / 3 and y_center < image_height / 3:
        corner = "Top-Right"
    elif x_center < image_width / 3 and y_center > 2 * image_height / 3:
        corner = "Bottom-Left"
    elif x_center > 2 * image_width / 3 and y_center > 2 * image_height / 3:
        corner = "Bottom-Right"
    else:
        corner = None

    position = f"{vertical}-{horizontal}"
    if corner:
        position = corner  # Override with corner if applicable

    return position

def generate_speech(all_text):
    """Generate a single speech audio file for all detected objects."""
    engine = pyttsx3.init()  # Initialize the TTS engine
    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)
    
    # Create a temporary file to save the audio
    with tempfile.NamedTemporaryFile(delete=False) as audio_file:
        audio_file_path = audio_file.name + ".mp3"
        engine.save_to_file(all_text, audio_file_path)  # Save to temporary file
        engine.runAndWait()  # Run the engine to generate speech
    return audio_file_path

# Streamlit UI
def home_page():
    st.title("Image-Centric Indoor Navigation Solution for Visually Impaired People")
    st.write("This app detects objects in images, analyzes their positions (left, right, top, bottom, center), and estimates their distance. The results are shown with bounding boxes around detected objects and audio output describing the detection.")
    st.write("Use the sidebar to upload an image and start detection.")

# Object Detection Page
def detection_page():
    st.title("Object Detection with Position Analysis and Voice Output")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "jfif"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect objects
        with st.spinner("Detecting objects..."):
            results = detect_objects(image)
        
        # Extract image dimensions
        image_width, image_height = image.size

        # Draw bounding boxes and determine positions
        detections = results[0].boxes  # Access the detection boxes from the results
        draw = ImageDraw.Draw(image)
        all_text = ""  # Initialize variable to collect all speech text

        st.write("### Detected Objects:")

        for det in detections:
            x_min, y_min, x_max, y_max = det.xyxy[0]  # Get the bounding box in xyxy format
            confidence = det.conf[0]  # Get the confidence score
            class_id = int(det.cls[0])  # Get the class ID
            label = model.names[class_id]  # Get the label from the class ID (from model.names)

            bbox = [x_min, y_min, x_max, y_max]
            position = analyze_position(bbox, image_width, image_height)

            # Estimate distance using bounding box area (simple heuristic)
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            distance = max(1, 1000 / np.sqrt(area))  # Example: Larger objects are closer

            # Prepare speech text for the object
            speech_text = f"Detected a {label} at position {position}. Estimated distance: {distance:.2f} cm.\n"
            all_text += speech_text  # Append each object's speech text

            # Print object details in the app
            st.write(f"**{label}** - Confidence: {confidence:.2f}, Position: {position}, Approximate Distance: {distance:.2f} cm")

            # Adjust bounding box thickness and color based on object type
            if label == "person":
                box_thickness = 8  # Thicker box for person
                box_color = "blue"
            elif label == "car":
                box_thickness = 8  # Slightly thinner box for car
                box_color = "green"
            else:
                box_thickness = 8  # Normal box for other objects
                box_color = "red"

            font = ImageFont.load_default()  # Default font, you can replace with a custom one

            # Draw bounding box with custom thickness and label in clear medium font
            draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=box_thickness)
            draw.text((x_min, y_min), f"{label} ({confidence*100:.1f}%)", fill=box_color, font=font)

        # Generate and play the single audio file for all objects
        audio_path = generate_speech(all_text)
        
        # Play the audio of the speech
        st.audio(audio_path, format="audio/mp3")

        # Display image with bounding boxes
        st.image(image, caption="Detected Objects with Bounding Boxes", use_column_width=True)

# Streamlit Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Object Detection"])

if page == "Home":
    home_page()
else:
    detection_page()


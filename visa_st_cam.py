import streamlit as st
import cv2
import pytesseract
from ultralytics import YOLO
from PIL import Image
import numpy as np
import re

# Path to your trained YOLO model
model_path = 'best.pt'

# Initialize the YOLO model
model = YOLO(model_path)

# Function to clean and format full name (all uppercase, no symbols)
def clean_full_name(text):
    text = re.sub(r'[^A-Z\s]', '', text)  # Remove non-uppercase letters and non-whitespace characters
    return ' '.join(text.split())  # Normalize spaces

# Function to clean and format passport number (alphanumeric, no symbols, uppercase)
def clean_passport_number(text):
    text = re.sub(r'[^A-Z0-9]', '', text)  # Remove non-alphanumeric characters
    return text.upper()  # Convert to uppercase just in case

# Function to format the date (either dd/mm/yyyy or yyyy/mm/dd)
def clean_date(text):
    # Regex patterns for both formats
    date_pattern_1 = re.compile(r'\b\d{2}/\d{2}/\d{4}\b')  # Pattern for dd/mm/yyyy
    date_pattern_2 = re.compile(r'\b\d{4}/\d{2}/\d{2}\b')  # Pattern for yyyy/mm/dd

    # Match the correct date format
    match_1 = date_pattern_1.search(text)
    match_2 = date_pattern_2.search(text)

    if match_1:
        return match_1.group(0)  # Return the dd/mm/yyyy date
    elif match_2:
        return match_2.group(0)  # Return the yyyy/mm/dd date
    else:
        return text  # Return the original text if no match

# Streamlit UI
st.title("YOLO + Tesseract OCR Application")

# Let the user choose between uploading a file or taking a photo
option = st.radio("Choose an input method", ('Upload an image', 'Take a photo'))

uploaded_file = None

# Handling file upload or camera input based on user's selection
if option == 'Upload an image':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
elif option == 'Take a photo':
    uploaded_file = st.camera_input("Take a photo")

if uploaded_file is not None:
    # Convert the file to an opencv image
    if option == 'Take a photo':
        # For camera input, handle the image file from Streamlit's camera
        image = Image.open(uploaded_file)
        image = np.array(image)
    else:
        # For uploaded file, handle the file as usual
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

    # Display the image
    st.image(image, caption='Uploaded Image' if option == 'Upload an image' else 'Captured Photo', use_column_width=True)

    # Run inference with YOLO model
    results = model.predict(image)

    # Dictionary to hold the OCR results with labels
    ocr_results = {}

    # Iterate over each result (assuming single image inference)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        class_indices = result.boxes.cls.cpu().numpy().astype(int)  # Class indices
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        labels = result.names  # YOLO class names (automatic label detection)

        # Iterate over each detected object (bounding box)
        for i, (box, class_index, confidence) in enumerate(zip(boxes, class_indices, confidences)):
            x1, y1, x2, y2 = map(int, box)  # Get the coordinates of the bounding box
            cropped_region = image[y1:y2, x1:x2]  # Crop the image to the bounding box region

            # Convert cropped region to RGB (from BGR)
            cropped_region_rgb = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB)
            cropped_image_pil = Image.fromarray(cropped_region_rgb)

            # Apply OCR to the cropped region
            extracted_text = pytesseract.image_to_string(cropped_image_pil, config='--psm 6').strip()

            # Fetch the class name (label) directly from YOLO predictions
            label = labels[class_index]

            # Apply specific formatting conditions based on label
            if label == 'full_name':
                extracted_text = clean_full_name(extracted_text)
            elif label == 'passport_no':
                extracted_text = clean_passport_number(extracted_text)
            elif label == 'date_of_issue' or label == 'date_of_expiry':
                extracted_text = clean_date(extracted_text)

            # Append or set text in dictionary to prevent overwriting
            if label in ocr_results:
                ocr_results[label] += ' ' + extracted_text
            else:
                ocr_results[label] = extracted_text

    # Display OCR results
    st.write("OCR Results:")
    for label, text in ocr_results.items():
        st.write(f"{label}: {text}")

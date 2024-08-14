import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
import cv2
import numpy as np
import re
from pdf2image import convert_from_bytes
import io
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
import pytesseract
from pytesseract import Output
import imutils
import os

credentials = service_account.Credentials.from_service_account_file(
    'ds-tech-402606-e89e06e3f36e.json'
)
client = vision.ImageAnnotatorClient(credentials=credentials)

UPLOAD_DIR = './aadhaar-masking-uploads'
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    
if 'show_bg_image' not in st.session_state:
    st.session_state['show_bg_image'] = True
    
def rotate_image(image, angle):
    rotated_image = imutils.rotate_bound(image, angle)
    return rotated_image

def get_image_orientation(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
        return results['rotate']
    except pytesseract.TesseractError as e:
        print(f"Tesseract failed with error: {e}")
        return None 
    
def hough_transforms(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = canny(thresh)
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)
    accum, angles, dists = hough_line_peaks(h, theta, d)
    return accum, angles, dists

def east(image, args):
    net = cv2.dnn.readNet(args['east_model_path'])
    blob = cv2.dnn.blobFromImage(image, 1.0, (args['width'], args['height']),
                                 (123.68, 116.78, 103.94), True, False)
    net.setInput(blob)
    scores, geometry = net.forward(['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])
    
    (numRows, numCols) = scores.shape[2:4]
    confidences = []
    angles = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        anglesData = geometry[0, 4, y]
        for x in range(numCols):
            if scoresData[x] < args['confidence_threshold']:
                continue
            angle = anglesData[x]
            confidences.append(scoresData[x])
            angles.append(angle)

    east_angle = np.median(np.rad2deg(angles))
    return image, east_angle

def east_hough_line(image, args):
    image, east_angle = east(image, args)
    h, theta, d = hough_transforms(image)
    theta = np.rad2deg(np.pi / 2 - theta)
    
    margin = args['margin_tolerance']
    low_thresh = east_angle - margin
    high_thresh = east_angle + margin
    
    filter_theta = theta[(theta > low_thresh) & (theta < high_thresh)]
    
    if len(filter_theta) == 0:
        return image, east_angle
    
    refined_angle = np.median(filter_theta)
    return image, refined_angle

def rotate_image_hough(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def mask_aadhaar_in_image(image_path, args):
    image = cv2.imread(image_path)
    angle_to_rotate = get_image_orientation(image)
    if angle_to_rotate is not None:
        rotated_image = rotate_image(image, angle_to_rotate)
    else:
        rotated_image = image

    tesseract_rotated_image_path = 'tesseract_result.jpg'
    cv2.imwrite(tesseract_rotated_image_path, rotated_image)
    processed_image, refined_angle = east_hough_line(rotated_image, args)
    
    corrected_image = rotate_image_hough(processed_image, -refined_angle)
    corrected_image_path = os.path.join(UPLOAD_DIR, 'corrected_image.jpg')
    cv2.imwrite(corrected_image_path, corrected_image)

    with open(corrected_image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    text_annotations = response.text_annotations

    if text_annotations:
        full_text = text_annotations[0].description
        all_matches = re.findall(r'\b\d{4} \d{4} \d{4}\b(?: \d{4})?', full_text)
        matches = [match for match in all_matches if len(match.replace(" ", "")) == 12]

        if matches:
            aadhaar_numbers = [match.replace(" ", "") for match in matches]
            char_list = []
            for annotation in text_annotations[1:]:
                for i, char in enumerate(annotation.description):
                    char_list.append((char, annotation.bounding_poly.vertices, i))

            color = (0, 0, 0)
            thickness = -1
            masked_coordinates = []

            for aadhaar_number in aadhaar_numbers:
                digit_coords = []
                seq_len = len(aadhaar_number)
                for i in range(len(char_list) - seq_len + 1):
                    sequence = char_list[i:i + seq_len]
                    sequence_text = ''.join([char[0] for char in sequence])
                    if sequence_text == aadhaar_number:
                        digit_coords = [char[1] for char in sequence]
                        if digit_coords not in masked_coordinates:
                            masked_coordinates.append(digit_coords)
                            break

                if digit_coords:
                    first_digit_coords = digit_coords[0]
                    eighth_digit_coords = digit_coords[7]

                    x_min = min(vertex.x for vertex in first_digit_coords)
                    y_min = min(vertex.y for vertex in first_digit_coords)
                    x_max = max(vertex.x for vertex in eighth_digit_coords)
                    y_max = max(vertex.y for vertex in eighth_digit_coords)

                    start_point = (x_min, y_min)
                    end_point = (x_max, y_max)
                    cv2.rectangle(corrected_image, start_point, end_point, color, thickness)
                else:
                    print(f"Aadhaar number sequence {aadhaar_number} not found in the image.")

            masked_image_path = os.path.join(UPLOAD_DIR, 'final_masked_image.jpg')
            cv2.imwrite(masked_image_path, corrected_image)
            os.remove(tesseract_rotated_image_path)
            os.remove(corrected_image_path)
            return masked_image_path
        else:
            print("Aadhaar number not found in the image.")
            return None
    else:
        print("No text found in the image.")
        return None

def display():
    st.markdown(
        """
        <style>
        [data-testid="stSidebarContent"] {
            color: black;
            background-color: #ffdd99;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
        <div style="text-align: right; margin-top: 10px;">
            <button class="fill" style="background-color: #FF8247; border: none; color: white; padding: 6px 12px; text-align: center; text-decoration: none; display: inline-block; font-size: 12px; margin: 4px 2px; border-radius: 12px; transition: background-color 0.5s, color 0.5s;">
                Beta v0.0.1
            </button>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: right; margin-top: 10px;">
            <a href="https://drive.google.com/drive/folders/1QUY_Bl-ArNc1OWW9JU3nWkWGKzpReXF-?usp=sharing" target="_blank">
                <button class="fill" style="--hover: #45a049; background-color: #4CAF50; border: none; color: white; padding: 6px 12px; text-align: center; text-decoration: none; display: inline-block; font-size: 12px; margin: 4px 2px; cursor: pointer; border-radius: 12px; transition: background-color 0.3s, color 0.3s;">
                    Get Test Documents
                </button>
            </a>
        </div>
        <style>
            .fill:hover,
            .fill:focus {
            box-shadow: inset 0 0 0 2em var(--hover);
            background-color: #45a049;
            color: #ffffff;
            transform: scale(1.05);
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("AADHAAR MASKING")

    st.image("aadhaar-masking-bg_new.png", caption="Mask your aadhaar cards with Doku-Wise!")

    st.sidebar.title('Upload Files Here')
    uploaded_files = st.sidebar.file_uploader("Choose Image or PDF files", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

    args = {
        'east_model_url': 'https://drive.google.com/uc?export=download&id=1pnkMiGjwyFDJmchGXpuvn7AO2-nL0PtQ',
        'east_model_path': 'frozen_east_text_detection.pb',
        'width': 320,
        'height': 320,
        'confidence_threshold': 0.5,
        'margin_tolerance': 10
    }


    if st.sidebar.button('Submit'):
        if uploaded_files:
            with st.spinner('Processing the file(s) uploaded...'):
                for idx, uploaded_file in enumerate(uploaded_files):
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())

                    if uploaded_file.type == "application/pdf":
                        with open(file_path, "rb") as pdf_file:
                            file_bytes = pdf_file.read()
                        images = convert_from_bytes(file_bytes)
                        for i, image in enumerate(images):
                            image_path = os.path.join(UPLOAD_DIR, f'page_{i}.jpg')
                            image.save(image_path, 'JPEG')
                            masked_image_path = mask_aadhaar_in_image(image_path, args)
                            if masked_image_path:
                                col1, col2 = st.columns([2, 3])
                                col1.image(image_path, caption='Original Image')
                                col2.image(masked_image_path, caption='Masked Image')
                    else:
                        masked_image_path = mask_aadhaar_in_image(file_path, args)
                        if masked_image_path:
                            col1, col2 = st.columns([2, 3])
                            col1.image(file_path, caption='Original Image')
                            col2.image(masked_image_path, caption='Masked Image')
                    
                    os.remove(file_path)        
                

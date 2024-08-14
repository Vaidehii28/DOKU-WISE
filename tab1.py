import streamlit as st
import torch
import json
import html
import torch.nn as nn
from io import BytesIO
import base64
# from PyPDF2 import PdfReader
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration
import os
import io
from pdf2image import convert_from_bytes
import plotly.express as px # type: ignore
import pandas as pd
from utilities.functions import extract_text_from_image, generate_predictions_for_14, parse_extracted_fields, get_language_name
from utilities.config import client
import gdown
import zipfile


def download_and_extract_model():
    # Google Drive link for the zipped folder (replace with your actual zip file link)
    url = "https://drive.google.com/uc?id=1AwrHyDVGxTAf7g9IE5JRoXaM_IdkOIoX"
    output = "t5_model.zip"
    
    # Check if the model folder already exists, otherwise download and extract it
    if not os.path.exists("t5_small-LATEST-CHANGES-400"):
        print("Downloading T5 model folder as zip...")
        gdown.download(url, output, quiet=False)
        
        # Extract the zip file
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        print("Model extracted.")
    else:
        print("Model folder already exists.")
    
    return "t5_small-LATEST-CHANGES-400"

# Initialize models and tokenizers
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

model_folder_path = download_and_extract_model()
t5_tokenizer = T5Tokenizer.from_pretrained(model_folder_path)
t5_model = T5ForConditionalGeneration.from_pretrained(model_folder_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Multimodal classifier definition
class MultimodalClassifier(nn.Module):
    def __init__(self, bert_model, inception_model, num_classes):
        super(MultimodalClassifier, self).__init__()
        self.bert_model = bert_model
        self.inception_model = inception_model
        self.fc = nn.Linear(bert_model.config.hidden_size + 2048, num_classes)
    
    def forward(self, text_input, image_input):
        text_output = self.bert_model(**text_input).pooler_output
        image_output = self.inception_model(image_input)
        if isinstance(image_output, tuple):
            image_output = image_output[0]
        combined_output = torch.cat((text_output, image_output.flatten(1)), dim=1)
        return self.fc(combined_output)

def download_model():
    url = "https://drive.google.com/uc?id=1ekEoDNm0gl2zuI7c4SjNQuhesIIrMpTz"
    
    model_path = "multimodal_model_15cats-new.pth"
    
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
    else:
        print("Model already exists.")
    
    return model_path

model_path = download_model()
num_classes = 15
inception_model = inception_v3(pretrained=True)
inception_model.fc = nn.Identity()
multimodal_model = MultimodalClassifier(bert_model, inception_model, num_classes)
multimodal_model.load_state_dict(torch.load(model_path))
multimodal_model.eval()


def tokenize_texts(texts):
    return bert_tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

def display():
    # Create a custom sidebar for Tab 1
    st.html(
        """
        <style>
        [data-testid="stSidebarContent"] {
            color: black;
            background-color: #d6d6ff;
        }
        </style>
        """
    )
    
    st.sidebar.markdown("""
        <div style="border: 2px solid #e6e6e6; padding: 10px; border-radius: 10px; background-color: #f9f9f9;">
            <h3 style="margin-top: 0; text-align: center; color: #333;">ACCEPTED DOCUMENTS:</h3>
            <ol style="list-style-type: decimal; padding-left: 20px; text-align: left; margin: 0;margin-top: -10px;">
                <li style="margin: 5px 0; font-size: 16px; color: #333;">Aadhaar-masked</li>
                <li style="margin: 5px 0; font-size: 16px; color: #333;">Aadhaar-unmasked</li>
                <li style="margin: 5px 0; font-size: 16px; color: #333;">Driving License</li>
                <li style="margin: 5px 0; font-size: 16px; color: #333;">Food/Drug License</li>
                <li style="margin: 5px 0; font-size: 16px; color: #333;">GST-Registration Certificate</li>
                <li style="margin: 5px 0; font-size: 16px; color: #333;">Import-Export Certificate</li>
                <li style="margin: 5px 0; font-size: 16px; color: #333;">Income Tax Return</li>
                <li style="margin: 5px 0; font-size: 16px; color: #333;">PAN</li>
                <li style="margin: 5px 0; font-size: 16px; color: #333;">Passport</li>
                <li style="margin: 5px 0; font-size: 16px; color: #333;">Rent Agreement</li>
                <li style="margin: 5px 0; font-size: 16px; color: #333;">Shop Establishment Act</li>
                <li style="margin: 5px 0; font-size: 16px; color: #333;">Udyam</li>
                <li style="margin: 5px 0; font-size: 16px; color: #333;">Utility Bills</li>
                <li style="margin: 5px 0; font-size: 16px; color: #333;">Voter ID</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="text-align: right; margin-top: 10px;">
            <button class="fill" style="hover: #E34800 ; background-color: #FF8247 ; border: none; color: white; padding: 6px 12px; text-align: center; text-decoration: none; display: inline-block; font-size: 12px; margin: 4px 2px; border-radius: 12px; transition: background-color 0.5s, color 0.5s;">
                Beta v0.0.1
            </button>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .stApp > div {
            background-color: #f7f7f7 !important; /* Orange */
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title('CLASSIFIER AND EXTRACTOR')
    st.markdown("""
        <div style="text-align: right; margin-top: 10px;">
            <a href="https://drive.google.com/drive/folders/1E8AGfBp3cl4u3FA-t3X6Y_rXUvm0XKOO" target="_blank">
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

    st.sidebar.title('Upload Files Here')
    uploaded_files = st.sidebar.file_uploader("Choose Image or PDF files", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

    st.image("CATEGORIES_BG.png", caption="Supported Documents")

    details_list = []
    final_pdf_details = []
    details_list_pdf = []
    num_images = 0
    num_pdfs = 0

    if st.sidebar.button('Submit'):
        with st.spinner('Processing the file(s) uploaded...'):
            if uploaded_files:
                num_images = len([file for file in uploaded_files if file.type.startswith('image')])
                print("NUM IMAGES: ", num_images)
                num_pdfs = len([file for file in uploaded_files if file.type == "application/pdf"])
                pdf_details_list = []
                if num_images > 3:
                    st.markdown(
                        '<div style="background-color: #ff9524; color: white; padding: 10px; border-radius: 5px;">'
                        f'{num_images} images uploaded, Extraction in progress.....'
                        '</div>',
                        unsafe_allow_html=True
                    )
                if num_pdfs > 3:
                    st.markdown(
                        f'<div style="background-color: #ff9524; color: white; padding: 10px; border-radius: 5px;">'
                        f'Processing {num_pdfs} PDF files...'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    pdf_details = []
                    
                    for uploaded_file in uploaded_files:
                        if uploaded_file.type == "application/pdf":
                            file_path = f"data/uploads/{uploaded_file.name}"
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            images = convert_from_bytes(open(file_path, 'rb').read())
                            os.remove(file_path)

                            for i, image in enumerate(images):
                                image_content = io.BytesIO()
                                image.save(image_content, format='JPEG')
                                extracted_text = extract_text_from_image(client, image_content.getvalue())
                                text_input = tokenize_texts(extracted_text)
                                image_input = transform(image).unsqueeze(0)

                                with torch.no_grad():
                                    outputs = multimodal_model(text_input, image_input)
                                    probabilities = torch.softmax(outputs, dim=1)
                                    predicted_class_idx = torch.argmax(outputs).item()

                                classes = ['Aadhaar-masked', 'Aadhaar-unmasked', 'PAN', 'DrivingLicense','Food-DrugLicenseCertificate', 'GST-Registration-Certificate', 'Import-Export-Certificate','IncomeTaxReturn', 'Passport', 'Rent-Agreement','Shop-Establishment-Act', 'VoterId', 'Udyam','UtilityBills', 'Other']
                                doc_type = classes[predicted_class_idx]
                                prediction = generate_predictions_for_14(t5_model, t5_tokenizer, doc_type, extracted_text)
                                prediction_dict = {}
                                for pair in prediction.split(";"):
                                    parts = pair.strip().split(": ")
                                    if len(parts) > 2:
                                        key = parts[0]
                                        value = ": ".join(parts[1:])
                                    else:
                                        key, value = parts
                                    prediction_dict[key.strip()] = value.strip()

                                prediction_json = json.dumps(prediction_dict) 
                                sanitized_extracted_text = html.escape(extracted_text)
                                
                                pdf_details.append({
                                    "Filename": uploaded_file.name,
                                    "Page number": i+1,
                                    "Extracted text": sanitized_extracted_text,
                                    "Predicted class": doc_type,
                                    "Extracted fields": prediction_json
                                })

                    details_list_pdf.extend(pdf_details)

                    if details_list_pdf:
                        st.subheader("Uploaded PDF documents summary:")
                        final_pdf_details = pd.DataFrame(details_list_pdf)
                        df_html = final_pdf_details.to_html(index=False, escape=False, classes='styled-table')
                        st.markdown("""
                            <style>
                                .styled-table {
                                    width: 100%; /* Use full width */
                                    border-collapse: collapse;
                                    margin: 10px 0;
                                    margin-top: -10px;
                                }
                                .styled-table th, .styled-table td {
                                    padding: 8px 10px; /* Adjust padding for better readability */
                                    text-align: left;
                                    border-bottom: 1px solid #ddd;
                                    max-width: 150px; /* Set a maximum width for table cells */
                                    word-wrap: break-word; /* Allow text to wrap within cells */
                                    white-space: nowrap; /* Prevent text wrapping to ensure each cell has a fixed height */
                                    overflow-x: auto; /* Enable horizontal scrolling if needed */
                                }
                                .styled-table th {
                                    background-color: #1E90FF; /* Dark shade of blue for header */
                                    color: #fff;
                                }
                                .styled-table tr:nth-child(even) {
                                    background-color: #f2f2f2; /* Light grey background for even rows */
                                }
                                .styled-table tr:nth-child(odd) {
                                    background-color: #ffffff; /* Light white background for odd rows */
                                }
                                .styled-table tr:hover {
                                    background-color: #ddd; /* Hover effect */
                                }
                            </style>
                        """, unsafe_allow_html=True)
                        st.write(df_html, unsafe_allow_html=True)
                        csv_filename = "pdf_ovd_summary"
                        csv_file = final_pdf_details.to_csv(index=False).encode('utf-8-sig')
                        b64_csv = base64.b64encode(csv_file).decode()
                        href = f'data:file/csv;base64,{b64_csv}'
                        st.markdown(f'<a href="{href}" download="{csv_filename}.csv" style="--hover: #45a049; background-color: #4CAF50; border: none; color: white; padding: 6px 12px; text-align: center; text-decoration: none; display: inline-block; font-size: 12px; margin: 10px 0 0 10px; cursor: pointer; border-radius: 12px; transition: background-color 0.3s, color 0.3s;">Download CSV</a>', unsafe_allow_html=True)

                        
                
                for uploaded_file in uploaded_files:
                    if uploaded_file.type == "application/pdf" and num_pdfs <= 3:
                        file_path = f"data/uploads/{uploaded_file.name}"
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        if uploaded_file.type == "application/pdf":
                            images = convert_from_bytes(open(file_path, 'rb').read())
                            os.remove(file_path)
                            for i, image in enumerate(images):
                                page_number = f"Page {i+1}"
                                image_content = io.BytesIO()
                                image.save(image_content, format='JPEG')
                                extracted_text = extract_text_from_image(client, image_content.getvalue())
                                text_input = tokenize_texts(extracted_text)
                                image_input = transform(image).unsqueeze(0)
                                with torch.no_grad():
                                    outputs = multimodal_model(text_input, image_input)
                                    probabilities = torch.softmax(outputs, dim=1)
                                    predicted_class_idx = torch.argmax(outputs).item()
                                classes = ['Aadhaar-masked', 'Aadhaar-unmasked', 'PAN', 'DrivingLicense','Food-DrugLicenseCertificate', 'GST-Registration-Certificate', 'Import-Export-Certificate','IncomeTaxReturn', 'Passport', 'Rent-Agreement','Shop-Establishment-Act', 'VoterId', 'Udyam','UtilityBills', 'Other']
                                doc_type = classes[predicted_class_idx]
                                prediction = generate_predictions_for_14(t5_model, t5_tokenizer, doc_type, extracted_text)
                                print("PREDICTION MADE: ", prediction)
                                # Convert prediction to JSON format
                                prediction_dict = {}
                                for pair in prediction.split(";"):
                                    if ": " in pair:
                                        key, value = pair.strip().split(": ", 1)
                                        prediction_dict[key.strip()] = value.strip()
                                    else:
                                        print(f"Skipping invalid pair: {pair}")
                                prediction_json = json.dumps(prediction_dict)
                                parsed_fields = parse_extracted_fields(prediction_json)
                                # Create DataFrame and convert to HTML
                                df = pd.DataFrame(parsed_fields)
                                df_html = df.to_html(index=False, escape=False, classes='styled-table')

                                # Display details
                                st.markdown("""
                                    <style>
                                        .styled-table {
                                            width: 100%;
                                            border-collapse: collapse;
                                            margin: 10px 0;
                                        }
                                        .styled-table th, .styled-table td {
                                            padding: 6px 7.5px;
                                            text-align: left;
                                            border-bottom: 1px solid #ddd;
                                        }
                                        .styled-table th {
                                            background-color: #1E90FF;
                                            color: #fff;
                                        }
                                        .styled-table tr:nth-child(even) {
                                            background-color: #f2f2f2;
                                        }
                                        .styled-table tr:nth-child(odd) {
                                            background-color: #ffffff;
                                        }
                                        .styled-table tr:hover {
                                            background-color: #ddd;
                                        }
                                    </style>
                                """, unsafe_allow_html=True)
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    with st.container():
                                        filename_text = f"Filename: {uploaded_file.name} ({page_number})"
                                        tile_style = f"""
                                            <p style="font-size: 15px; font-weight: bold; margin: 0;">{filename_text}</p>
                                        """
                                        st.markdown(tile_style, unsafe_allow_html=True)
                                        st.image(image, caption='Uploaded Image.', use_column_width=True)
                                        st.markdown("</div>", unsafe_allow_html=True)
                                with col2:
                                    sanitized_extracted_text = html.escape(extracted_text)
                                    container_style = f"""
                                        <div style="
                                            border-radius: 10px;
                                            padding: 20px;
                                            margin-bottom: 20px;
                                            background-color: #ffffff;
                                            border: 2px solid #f4f4f4;
                                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                                        ">
                                            <h3 style="color: #333; margin-top: 0;">Extracted Text:</h3>
                                            <div style="height: 160px; overflow-y: scroll; padding: 10px; background-color: #ffffff; border-radius: 5px; color: #333;">
                                                {sanitized_extracted_text}
                                            </div>
                                            <h3 style="color: #333;">Predicted Class:</h3>
                                            <div style="background-color: #00c04b; padding: 10px; color: #ffffff; margin-bottom: 20px;">
                                                {doc_type}
                                            </div>
                                            <h3 style="color: #333;">Extracted Fields:</h3>
                                            <div style="height: 160px; overflow-y: scroll; border-radius: 5px;border: 1px solid #D8FFB1; color: #0f5132;">
                                                {df_html}
                                            </div>
                                        </div>
                                    """
                                    st.markdown(container_style, unsafe_allow_html=True)

                                st.subheader("Probabilities:")
                                probabilities_dict = {classes[i]: probabilities[0][i].item() for i in range(len(classes))}
                                df_prob = pd.DataFrame(list(probabilities_dict.items()), columns=['Class', 'Probability'])
                                fig = px.bar(df_prob, x='Probability', y='Class', color='Class', orientation='h', title='Probability Distribution')
                                fig.update_layout(
                                    height=400,
                                    margin=dict(l=20, r=20, t=50, b=20),
                                    title=dict(font=dict(size=18)),
                                    yaxis=dict(tickfont=dict(size=12)),
                                    xaxis=dict(tickfont=dict(size=12))
                                )
                                st.plotly_chart(fig)
                                st.markdown("<hr>", unsafe_allow_html=True)
                        
                    else:
                        file_path = f"data/uploads/{uploaded_file.name}"
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        image = Image.open(file_path).convert('RGB')
                        extracted_text = extract_text_from_image(client, open(file_path, 'rb').read())
                        text_input = tokenize_texts(extracted_text)
                        image_input = transform(image).unsqueeze(0)

                        with torch.no_grad():
                            outputs = multimodal_model(text_input, image_input)
                            probabilities = torch.softmax(outputs, dim=1)
                            predicted_class_idx = torch.argmax(outputs).item()

                        classes = ['Aadhaar-masked', 'Aadhaar-unmasked', 'PAN', 'DrivingLicense','Food-DrugLicenseCertificate', 'GST-Registration-Certificate', 'Import-Export-Certificate','IncomeTaxReturn', 'Passport', 'Rent-Agreement','Shop-Establishment-Act', 'VoterId', 'Udyam','UtilityBills', 'Other']
                        doc_type = classes[predicted_class_idx]
                        prediction = generate_predictions_for_14(t5_model, t5_tokenizer, doc_type, extracted_text)
                        if prediction is None:
                            prediction_json = json.dumps({"message": "No fields to extract"})
                        else:
                            prediction_dict = {}
                            for pair in prediction.split(";"):
                                parts = pair.strip().split(": ")
                                if len(parts) < 2:
                                    continue
                                if len(parts) > 2:
                                    key = parts[0]
                                    value = ": ".join(parts[1:])
                                else:
                                    key, value = parts
                                    prediction_dict[key.strip()] = value.strip()
                        prediction_json = json.dumps(prediction_dict) 
                        detected_language = get_language_name(extracted_text)
                                
                        sanitized_extracted_text = html.escape(extracted_text)
                        file_details = {
                            "Image Name": uploaded_file.name,
                            "Extracted Text": sanitized_extracted_text,
                            "Detected Language": detected_language,
                            "Predicted Class": doc_type,
                            "Extracted Fields": prediction_json,
                            "Image": image
                        }

                        details_list.append(file_details)

            if num_images > 5:
                st.subheader("Uploaded images summary:")
                details_df = pd.DataFrame(details_list)
                details_df = details_df.drop(columns=['Image'], axis=1)

                df_html = details_df.to_html(index=False, escape=False, classes='styled-table')
                st.markdown("""
                    <style>
                        .styled-table {
                            width: 100%; /* Use full width */
                            border-collapse: collapse;
                            margin: 10px 0;
                            margin-top: -10px;
                        }
                        .styled-table th, .styled-table td {
                            padding: 8px 10px; /* Adjust padding for better readability */
                            text-align: left;
                            border-bottom: 1px solid #ddd;
                            max-width: 150px; /* Set a maximum width for table cells */
                            word-wrap: break-word; /* Allow text to wrap within cells */
                            white-space: nowrap; /* Prevent text wrapping to ensure each cell has a fixed height */
                            overflow-x: auto; /* Enable horizontal scrolling if needed */
                        }
                        .styled-table th {
                            background-color: #1E90FF; /* Dark shade of blue for header */
                            color: #fff;
                        }
                        .styled-table tr:nth-child(even) {
                            background-color: #f2f2f2; /* Light grey background for even rows */
                        }
                        .styled-table tr:nth-child(odd) {
                            background-color: #ffffff; /* Light white background for odd rows */
                        }
                        .styled-table tr:hover {
                            background-color: #ddd; /* Hover effect */
                        }
                    </style>
                """, unsafe_allow_html=True)
                st.write(df_html, unsafe_allow_html=True)
                csv_filename = "images_ovd_summary"
                csv_file = details_df.to_csv(index=False).encode('utf-8-sig')
                b64_csv = base64.b64encode(csv_file).decode()
                href = f'data:file/csv;base64,{b64_csv}'
                st.markdown(f'<a href="{href}" download="{csv_filename}.csv" style="--hover: #45a049; background-color: #4CAF50; border: none; color: white; padding: 6px 12px; text-align: center; text-decoration: none; display: inline-block; font-size: 12px; margin: 10px 0 0 10px; cursor: pointer; border-radius: 12px; transition: background-color 0.3s, color 0.3s;">Download CSV</a>', unsafe_allow_html=True)


            else:
                for details in details_list:
                    col1, col2 = st.columns([1, 1])

                    if details['Extracted Fields'] == '{"message": "No fields to extract"}':
                        parsed_fields = []
                    else:
                        parsed_fields = parse_extracted_fields(details['Extracted Fields'])

                    df = pd.DataFrame(parsed_fields)
                    df_html = df.to_html(index=False, escape=False, classes='styled-table')
                    st.markdown("""
                        <style>
                            .styled-table {
                                width: 100%;
                                border-collapse: collapse;
                                margin: 10px 0;
                            }
                            .styled-table th, .styled-table td {
                                padding: 6px 7.5px;
                                text-align: left;
                                border-bottom: 1px solid #ddd;
                            }
                            .styled-table th {
                                background-color: #1E90FF; /* Dark shade of blue for header */
                                color: #fff;
                            }
                            .styled-table tr:nth-child(even) {
                                background-color: #f2f2f2; /* Light grey background for even rows */
                            }
                            .styled-table tr:nth-child(odd) {
                                background-color: #ffffff; /* Light white background for odd rows */
                            }
                            .styled-table tr:hover {
                                background-color: #ddd; /* Hover effect */
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    with col1:
                        with st.container():
                            filename_text = f"Filename: {details['Image Name']}"
                            tile_style = f"""
                                <p style="font-size: 15px; font-weight: bold; margin: 0;">{filename_text}</p>
                            """
                            st.markdown(tile_style, unsafe_allow_html=True)
                            st.image(details['Image'], caption='Uploaded Image.', use_column_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                    with col2:
                        sanitized_extracted_text = html.escape(details['Extracted Text'])
                        sanitized_predicted_class = html.escape(details['Predicted Class'])
                        container_style = f"""
                            <div style="
                                border-radius: 10px;
                                padding: 20px;
                                margin-bottom: 20px;
                                background-color: #ffffff;
                                border: 2px solid #f4f4f4;
                                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                            ">
                                <h3 style="color: #333; margin-top: 0;">Extracted Text:</h3>
                                <div style="height: 160px; overflow-y: auto; padding: 10px; background-color: #ffffff; border-radius: 5px; color: #333; margin-bottom: 20px; border: 1px solid #e6e6e6;">
                                    {sanitized_extracted_text}
                                </div>
                                <h3 style="color: #333;">Detected Language:</h3>
                                <div style="background-color: #f9c846; padding: 10px; color: #ffffff; margin-bottom: 20px;">
                                    {details['Detected Language']}
                                </div>
                                <h3 style="color: #333;">Predicted Class:</h3>
                                <div style="background-color: #00c04b; padding: 10px; color: #ffffff; margin-bottom: 20px;">
                                    {sanitized_predicted_class}
                                </div>
                                <h3 style="color: #333;">Extracted Fields:</h3>
                                <div style="height: 160px; overflow-y: auto; color: #0f5132;">
                                    {df_html if df.shape[0] > 0 else '<p>No fields to extract.</p>'}
                                </div>
                            </div>
                        """
                        st.markdown(container_style, unsafe_allow_html=True)
                    st.subheader("Probabilities:")
                    probabilities_dict = {classes[i]: probabilities[0][i].item() for i in range(len(classes))}
                    df_prob = pd.DataFrame(list(probabilities_dict.items()), columns=['Class', 'Probability'])
                    fig = px.bar(df_prob, x='Probability', y='Class', color='Class', orientation='h', title='Probability Distribution')
                    fig.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=50, b=20),
                        title=dict(font=dict(size=18)),
                        yaxis=dict(tickfont=dict(size=12)),
                        xaxis=dict(tickfont=dict(size=12))
                    )
                    st.plotly_chart(fig)
                    st.markdown("<hr>", unsafe_allow_html=True)

    else:
        st.sidebar.warning("Please upload a document.")
        

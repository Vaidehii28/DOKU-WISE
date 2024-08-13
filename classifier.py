#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
from google.cloud import vision
from tqdm import tqdm
import torchvision


# In[ ]:


# Set up Google Vision API
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/lendingkart/vaidehi.mahyavanshi/document-classifier/ds-tech-402606-e89e06e3f36e.json"

# Extract text from images using Google Vision API
def extract_text_and_save(image_path, save_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    extracted_text = texts[0].description if texts else ""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as text_file:
        text_file.write(extracted_text)
    
    return extracted_text


# In[ ]:


# Directory paths
dataset_path = "/home/lendingkart/vaidehi.mahyavanshi/document-classifier/final-classifier/classifier-8-cats/dataset-with-15-categories-new"
ocr_text_path = "/home/lendingkart/vaidehi.mahyavanshi/document-classifier/final-classifier/classifier-8-cats/ocr-text-15cats-new"
text_data = {}

# # Classes
classes = ['Aadhaar-masked', 'Aadhaar-unmasked', 'PAN', 'DrivingLicense','Food-DrugLicenseCertificate', 'GST-Registration-Certificate', 'Import-Export-Certificate','IncomeTaxReturn', 'Passport', 'Rent-Agreement','Shop-Establishment-Act', 'VoterId', 'Udyam','UtilityBills', 'Other']


# In[ ]:


# Extract text from images, save as .txt files, and save to JSON
print("Starting text extraction from images...")
for category in tqdm(classes, desc="Processing categories"):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        text_data[category] = []
        image_files = os.listdir(category_path)
        for image_name in tqdm(image_files, desc=f"Processing {category} images", leave=False):
            image_path = os.path.join(category_path, image_name)
            text_save_path = os.path.join(ocr_text_path, category, f"{os.path.splitext(image_name)[0]}.txt")
            text = extract_text_and_save(image_path, text_save_path)
            text_data[category].append({"image_path": image_path, "text": text, "label": category})
        print(f"Completed {category}: {len(image_files)} images processed.")

print("Saving extracted texts to JSON file...")
with open('extracted_texts_15cats-new.json', 'w') as f:
    json.dump(text_data, f)


# In[ ]:


# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


# In[ ]:


# Define text tokenizer function
def tokenize_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')


# In[ ]:


# Define image preprocessing function
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[ ]:


def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image)
    except OSError as e:
        print(f"Error loading image {image_path}: {e}")
        return None


# In[ ]:


# Load InceptionV3 model
from torchvision.models import inception_v3
inception_model = inception_v3(pretrained=True)
inception_model.fc = nn.Identity()  # Remove the final classification layer


# In[ ]:


# Define multimodal classifier
class MultimodalClassifier(nn.Module):
    def __init__(self, bert_model, inception_model, num_classes):
        super(MultimodalClassifier, self).__init__()
        self.bert_model = bert_model
        self.inception_model = inception_model
        self.fc = nn.Linear(bert_model.config.hidden_size + 2048, num_classes)
    
    def forward(self, text_input, image_input):
        text_output = self.bert_model(**text_input).pooler_output
        image_output = self.inception_model(image_input)
        if isinstance(image_output, torchvision.models.InceptionOutputs):
            image_output = image_output.logits
        print(f'image_output shape: {image_output.shape}')
        
        combined_output = torch.cat((text_output, image_output), dim=1)
        print(f'combined_output shape: {combined_output.shape}')
        
        return self.fc(combined_output)


# In[ ]:


# Create the multimodal classifier
num_classes = len(classes)
print("NUMBER OF CLASSES: ", num_classes)
multimodal_model = MultimodalClassifier(bert_model, inception_model, num_classes)


# In[ ]:


class DocumentDataset(Dataset):
    def __init__(self, data, classes):
        self.data = data
        self.labels = {label: idx for idx, label in enumerate(classes)}

    def __len__(self):
        return sum(len(self.data[label]) for label in self.data)

    def __getitem__(self, idx):
        current_idx = 0
        for label in self.data:
            if idx < current_idx + len(self.data[label]):
                item = self.data[label][idx - current_idx]
                text = item['text']
                image_path = item['image_path']
                label_idx = self.labels[label]
                text_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
                image_input = preprocess_image(image_path)
                if image_input is None:
                    # Skip the corrupted image by recursively calling __getitem__
                    return self.__getitem__((idx + 1) % self.__len__())
                return text_input, image_input, label_idx
            current_idx += len(self.data[label])



# In[ ]:


# Custom collate function to handle padding of text inputs
def custom_collate_fn(batch):
    text_inputs = [item[0] for item in batch]
    image_inputs = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])

    input_ids = pad_sequence([ti['input_ids'].squeeze() for ti in text_inputs], batch_first=True)
    attention_mask = pad_sequence([ti['attention_mask'].squeeze() for ti in text_inputs], batch_first=True)
    token_type_ids = pad_sequence([ti['token_type_ids'].squeeze() for ti in text_inputs], batch_first=True)
    text_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

    return text_inputs, image_inputs, labels


# In[ ]:


# Load data and prepare the dataset
with open('extracted_texts_15cats-new.json') as f:
    text_data = json.load(f)


# In[ ]:


dataset = DocumentDataset(text_data, classes)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)


# In[ ]:


# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(multimodal_model.parameters(), lr=1e-4)


# In[ ]:


# Training function with more print statements for debugging

def train(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (text_input, image_input, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            print(f'batch_idx: {batch_idx}, image_input shape: {image_input.shape}')
            outputs = model(text_input, image_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}')
    print('Training finished.')


# # # # In[ ]:


# Train the model
print("Starting training...")
train(multimodal_model, dataloader, criterion, optimizer, epochs=10)


# # In[ ]:


# # Save the model
torch.save(multimodal_model.state_dict(), 'multimodal_model_15cats-new.pth')
print("Model saved as 'multimodal_model_15cats-new.pth'.")


# In[ ]:


# Testing function with prediction and thresholding
def test(model, dataloader, criterion, threshold=0.5):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch_idx, (text_input, image_input, labels) in enumerate(dataloader):
            outputs = model(text_input, image_input)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probabilities = nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            print(f'Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Accuracy of the model on the test images: {accuracy}%')

    # Applying thresholding
    thresholded_predictions = []
    for prob in all_probabilities:
        max_prob = max(prob)
        if max_prob >= threshold:
            thresholded_predictions.append(int(prob.argmax()))
        else:
            thresholded_predictions.append(len(classes) - 1)  # Assigning "Other" class

    return all_labels, thresholded_predictions, all_probabilities


# In[ ]:


# Load the trained model for testing
multimodal_model.load_state_dict(torch.load('multimodal_model_15cats-new.pth'))
print("Model loaded for testing.")


# In[ ]:


# Prepare the test dataset and dataloader
test_dataset = DocumentDataset(text_data, classes)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)


# In[ ]:


# Define criterion (same as used during training)
criterion = nn.CrossEntropyLoss()


# In[ ]:


# Test the model
print("Starting testing...")
all_labels, all_predictions, all_probabilities = test(multimodal_model, test_dataloader, criterion)


# In[ ]:


# Confusion matrix and classification report
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# cm = confusion_matrix(all_labels, all_predictions)
# report = classification_report(all_labels, all_predictions, target_names=classes)
# print("Classification Report:\n", report)

# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.savefig('confusion_matrix_15.png')
# plt.show()
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Normalize the confusion matrix
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Generate classification report
report = classification_report(all_labels, all_predictions, target_names=classes)
print("Classification Report:\n", report)

# Plot the confusion matrix
plt.figure(figsize=(20, 16))
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Percentage)')
plt.savefig('confusion_matrix_15cats-new.png')
plt.show()



# In[ ]:


# Misclassifications
# misclassifications = []
# for i, (label, prediction, prob) in enumerate(zip(all_labels, all_predictions, all_probabilities)):
#     if label != prediction:
#         misclassifications.append((label, prediction, prob[label], prob[prediction], test_dataset.data[label][i]['image_path']))

# print("Misclassifications:")
# for true_label, predicted_label, true_prob, pred_prob, image_path in misclassifications[:5]:  # Show top 5 misclassifications
#     print(f"True label: {classes[true_label]}, Predicted: {classes[predicted_label]}, True prob: {true_prob:.4f}, Pred prob: {pred_prob:.4f}, Image: {image_path}")


# # In[ ]:


# # Visualize correct predictions
# correct_predictions = []
# for i, (label, prediction, prob) in enumerate(zip(all_labels, all_predictions, all_probabilities)):
#     if label == prediction:
#         image_path = test_dataset.data[i][1]  # Retrieve the image path directly using the index
#         correct_predictions.append((label, prediction, prob[label], prob[prediction], image_path))

# print("Correct Predictions:")
# for true_label, predicted_label, true_prob, pred_prob, image_path in correct_predictions[:25]:  # Show top 5 correct predictions
#     print(f"True label: {classes[true_label]}, Predicted: {classes[predicted_label]}, True prob: {true_prob:.4f}, Pred prob: {pred_prob:.4f}, Image: {image_path}")
#     image = Image.open(image_path)
#     plt.imshow(image)
#     plt.title(f"True: {classes[true_label]} | Pred: {classes[predicted_label]}")
#     plt.show()


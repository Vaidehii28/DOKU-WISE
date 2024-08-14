import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision.models import inception_v3
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration
import gdown
import os

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


def get_bert_tokenizer():
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return bert_tokenizer

def get_bert_model():
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    return bert_model

def get_t5_tokenizer():
    t5_tokenizer = T5Tokenizer.from_pretrained('t5_small-LATEST-CHANGES-400')
    return t5_tokenizer

def get_t5_model():
    t5_model = T5ForConditionalGeneration.from_pretrained('t5_small-LATEST-CHANGES-400')
    return t5_model

def download_model():
    url = "https://drive.google.com/uc?id=1ekEoDNm0gl2zuI7c4SjNQuhesIIrMpTz"
    
    model_path = "multimodal_model_15cats-new.pth"
    
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
    else:
        print("Model already exists.")
    
    return model_path

def get_multimodal_model():
    # Load multimodal model
    model_path = download_model()
    num_classes = 15
    inception_model = inception_v3(pretrained=True)
    inception_model.fc = nn.Identity()
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    multimodal_model = MultimodalClassifier(bert_model, inception_model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multimodal_model.load_state_dict(torch.load(model_path, map_location=device))
    multimodal_model.eval()
    return multimodal_model

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

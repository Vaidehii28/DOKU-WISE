#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import ast
import os
import logging
import pandas as pd
from datasets import load_dataset

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    T5Config
)


# In[ ]:


MODEL = 't5-small'
BATCH_SIZE = 2
NUM_PROCS = 4
EPOCHS = 400
OUT_DIR = 't5_small-LATEST-CHANGES-400'
MAX_LENGTH = 256
LEARNING_RATE = 0.0001
# MODEL = 't5-base' 

# BATCH_SIZE = 2
# NUM_PROCS = 4
# EPOCHS = 50
# OUT_DIR = 't5base_14CATS_adjusted'
# MAX_LENGTH = 256



os.makedirs(OUT_DIR, exist_ok=True)


# In[ ]:


# Load the CSV file
file_path = 'Dataset-latest-changes.csv'
df = pd.read_csv(file_path)


# In[ ]:


def is_valid_dict(s):
    try:
        d = ast.literal_eval(s)
        return isinstance(d, dict)
    except (ValueError, SyntaxError):
        return False

def clean_entry(s):
    try:
        s = s.replace("''", "'")
        d = ast.literal_eval(s)
        return d
    except (ValueError, SyntaxError):
        return None


# In[ ]:


# Check and clean the '2' column
df['2_valid'] = df['2'].apply(lambda x: is_valid_dict(x))
df['2_cleaned'] = df['2'].apply(lambda x: clean_entry(x) if is_valid_dict(x) else None)

df_cleaned = df.dropna(subset=['2_cleaned'])

df_cleaned = df_cleaned.drop(columns=['2_valid', '2_cleaned'])

cleaned_file_path = 'clean_FINAL_DATASET14-400.csv'
df_cleaned.to_csv(cleaned_file_path, index=False)


# In[ ]:


dataset = load_dataset(
    'csv',
    data_files=cleaned_file_path,
    split='train'
)

split = dataset.train_test_split(test_size=0.2)
dataset_train = split['train']
dataset_test = split['test']

print(f"Training samples: {len(dataset_train)}")
print(f"Testing samples: {len(dataset_test)}")
print(f"Example training data:\n{dataset_train[0]}")
print(f"Example testing data:\n{dataset_test[0]}")


# In[ ]:


# Initialize tokenizer
print("Initializing tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL)


# In[ ]:


# def create_prompt(doc_type, text):
#     prompts = {
#         'Aadhaar-masked': f"Extract Aadhaar number, Name, Gender, DOB, Address (if present): {text}",
#         'Aadhaar-unmasked': f"Extract Aadhaar number, Name, Gender, DOB, Address (if present): {text}",
#         'PAN': f"Extract Permanent Account Number, Name, Father name, Date of Birth, Date of Incorporation/Formation (if present): {text}",
#         'DrivingLicense': f"Extract DL No., Name, Address, DOB, Issue date, Validity NT, Validity TR (if present): {text}",
#         'Passport': f"Extract Passport No., SurName, Given Name(s), Date of Birth, Place of Birth, Issue date, Date of Expiry, Place of Issue: {text}",
#         'VoterId': f"Extract Voter Id number, Name, Gender, Father name, Husband name (if present), Date of Birth, Address (if present): {text}",
#         'Udyam': f"Extract Udyam Registration Number, Type of Enterprise: {text}",
#         'GST-Registration-Certificate': f"Extract Registration number, Legal Name, Trade Name, Constitution of Business, Address of Principal Place of Business, Date of Liability, Date of Validity, Type of Registration: {text}",
#         'Import-Export-Certificate': f"Extract IEC, PAN, Firm Name, Nature of Concern, Date of Issue, Registered Address: {text}",
#         'Rent-Agreement': f"Extract address, first party, second party, issue date/comencement date, and expiry date or period if expiry date is not mentioned: {text}",
#         'Food-DrugLicenseCertificate': f"Extract Registration number, License number, Name of Food Business Operator, Permanent address of Food Business Operator, Address of location where food business is to be conducted/premises, Kind of Business, Issued on, Valid upto: {text}",
#         'IncomeTaxReturn': f"Extract Assessment Year, PAN, Name, Address, Status, Form Number, e-filing acknowledgement number, Total income, Net tax payable, Taxes paid: {text}",
#         'Shop-Establishment-Act': f"Extract Registration number, Name of establishment, Address of establishment, Employer name, Employer address, Date of issue, Expiry date: {text}",
#         'UtilityBills': f"Extract Name, Address: {text}"
#     }

#     return prompts.get(doc_type, f"Extract information: {text}")

def create_prompt(doc_type, text):
    prompts = {
        'Aadhaar-masked': f"Extract Aadhaar number, Name, Gender, DOB, Address (if present): {text}",
        'Aadhaar-unmasked': f"Extract Aadhaar number, Name, Gender, DOB, Address (if present): {text}",
        'PAN': f"Extract Permanent Account Number, Name, Father name, Date of Birth, Date of Incorporation/Formation (if present): {text}",
        'DrivingLicense': f"Extract DL No., Name, Address, DOB, Issue date, Validity NT, Validity TR (if present): {text}",
        'Passport': f"Extract Passport No., SurName, Given Name(s), Date of Birth, Place of Birth, Issue date, Date of Expiry, Place of Issue: {text}",
        'VoterId': f"Extract Voter Id number, Name, Gender, Father name, Husband name (if present), Date of Birth, Address (if present): {text}",
        'Udyam': f"Extract Udyam Registration Number, Type of Enterprise/ Organization, Major Activity, SOCIAL CATEGORY OF ENTREPRENEUR, NAME OF UNIT(S), OFFICIAL ADDRESS OF ENTERPRISE, Flat/Door/Block No., Name of Premises/ Building, Village/Town, Block, Road/Street/Lane, City, State, District, Mobile, Email, DATE OF INCORPORATION / REGISTRATION OF ENTERPRISE, DATE OF COMMENCEMENT OF PRODUCTION/BUSINESS, NATIONAL INDUSTRY CLASSIFICATION CODE(S), DATE OF UDYAM REGISTRATION, Date of printing: {text}",
        'GST-Registration-Certificate': f"Extract Registration number, Legal Name, Trade Name, Constitution of Business, Address of Principal Place of Business, Period of Validity, Type of Registration, Date of Issue of Certificate, Details of Additional Places of Business: {text}",
        'Import-Export-Certificate': f"Extract IEC, PAN, Firm Name, Nature of Concern, Date of Issue, Registered Address, Issued From File No.: {text}",
        'Rent-Agreement': f"Extract address, first party, second party, issue date/comencement date, and expiry date or period if expiry date is not mentioned: {text}",
        'Food-DrugLicenseCertificate': f"Extract Registration number, License number, Name of Food Business Operator, Permanent address of Food Business Operator, Address of location where food business is to be conducted/premises, Kind of Business, Issued on, Valid upto: {text}",
        'IncomeTaxReturn': f"Extract Assessment Year, PAN, Name, Address, Status, Form Number, e-filing acknowledgement number, Total income, Net tax payable, Taxes paid: {text}",
        'Shop-Establishment-Act': f"Extract Registration number, Name of establishment, Address of establishment, Employer name, Employer address, Date of issue, Expiry date: {text}",
        'UtilityBills': f"Extract Name, Address: {text}"
    }

    return prompts.get(doc_type, f"Extract information: {text}")


# In[ ]:


def preprocess_function(examples):
    inputs = [create_prompt(doc_type, text) for doc_type, text in zip(examples['type_of_document'], examples['1'])]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length'
    )

    cleaned_tag = []
    for text in examples['2']:
        if isinstance(text, str):
            try:
                text = eval(text)
                cleaned_tag.append('; '.join([f"{k}: {v}" for k, v in text.items() if v is not None]))
            except SyntaxError:
                print(f"Skipping entry due to syntax error: {text}")
                cleaned_tag.append('')
            except Exception as e:
                print(f"Skipping entry due to unexpected error: {e}")
                cleaned_tag.append('')
        else:
            print(f"Skipping entry due to non-string text: {text}")
            cleaned_tag.append('')

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            cleaned_tag,
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length'
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[ ]:


# Tokenize the dataset
print("Tokenizing the dataset...")
tokenized_train = dataset_train.map(
    preprocess_function,
    batched=True,
    num_proc=NUM_PROCS
)

tokenized_test = dataset_test.map(
    preprocess_function,
    batched=True,
    num_proc=NUM_PROCS
)


# In[ ]:


# Initialize the T5 model for conditional generation
print("Initializing T5 model...")
config = T5Config.from_pretrained(MODEL, dropout_rate=0.1)
model = T5ForConditionalGeneration.from_pretrained(MODEL, config=config)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

logger = logging.getLogger(__name__)

class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        if state.is_local_process_zero:
            logger.info(f"Step {state.global_step}: {logs}")
            print(f"Step {state.global_step}: {logs}")


# In[ ]:

# Define training arguments
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_dir=OUT_DIR,
    logging_steps=10,
    evaluation_strategy='steps',
    save_steps=200,
    eval_steps=200,
    load_best_model_at_end=True,
    save_total_limit=5,
    report_to='tensorboard',
    learning_rate=LEARNING_RATE,
    fp16=True,
    dataloader_num_workers=4,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    callbacks=[LogCallback]
)

print("Training the model...")
trainer.train()


# In[ ]:


# Save the tokenizer and model
model.save_pretrained(OUT_DIR)
print("Saving the tokenizer...")
tokenizer.save_pretrained(OUT_DIR)


# In[ ]:


# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(OUT_DIR)
model = T5ForConditionalGeneration.from_pretrained(OUT_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[ ]:


# Function to preprocess the test data
def preprocess_test_data(examples):
    inputs = [create_prompt(doc_type, text) for doc_type, text in zip(examples['type_of_document'], examples['1'])]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length'
    )

    cleaned_tag = []
    for text in examples['2']:
        if isinstance(text, str):
            try:
                text = eval(text)
                cleaned_tag.append('; '.join([f"{k}: {v}" for k, v in text.items() if v is not None]))
            except SyntaxError:
                print(f"Skipping entry due to syntax error: {text}")
                cleaned_tag.append('')
            except Exception as e:
                print(f"Skipping entry due to unexpected error: {e}")
                cleaned_tag.append('')
        else:
            print(f"Skipping entry due to non-string text: {text}")
            cleaned_tag.append('')

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            cleaned_tag,
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length'
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[ ]:


# Preprocess the test data
tokenized_test = dataset_test.map(
    preprocess_test_data,
    batched=True,
    num_proc=NUM_PROCS
)


# In[ ]:


# Function to generate predictions
def generate_predictions(model, tokenizer, test_dataset):
    model.eval()
    predictions = []
    for example in test_dataset:
        input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)
        output_ids = model.generate(input_ids, max_length=MAX_LENGTH)
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred)
    return predictions


# In[ ]:


# Generate predictions on the test data
predictions = generate_predictions(model, tokenizer, tokenized_test)

num_examples = min(50, len(tokenized_test['1']))
for i in range(num_examples):
    print("="*50)
    print(f"Example {i+1}:")
    print(f"Input Text: {tokenized_test['1'][i]}")
    print(f"Predicted Output: {predictions[i]}")
    print(f"Actual Output: {tokenized_test['2'][i]}")
    print("="*50)


# In[ ]:


# Save predictions to a file
output_df = pd.DataFrame({
    'input': tokenized_test['1'],
    'actual_output': tokenized_test['2'],
    'predicted_output': predictions
})
output_df.to_csv('preds-LATEST-400.csv', index=False)

print("Predictions saved to 'preds-LATEST-400.csv'")


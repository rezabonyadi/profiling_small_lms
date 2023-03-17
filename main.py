import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from language_models import ft5_models

import numpy as np

# Define the available models and their corresponding model names
models = {
    "Flan T5 small": "google/flan-t5-small",
    "Flan T5 base": "google/flan-t5-base",
    "Flan T5 large": "google/flan-t5-large",
    # "Flan T5 xl": "google/flan-t5-xl",    
}

# Create a dropdown to select the model
model_name = st.sidebar.selectbox("Select a model", list(models.keys()))
max_in = st.sidebar.text_input("max input length", "512")
max_out = st.sidebar.text_input("max output length", "64")
temprature = st.sidebar.text_input("temprature", "0.7")

device = st.sidebar.selectbox("Device", ['cpu', 'cuda:0'])

# Create input fields for the context and task
context = st.text_area("Context")
task = st.text_input("Task")

# # Load the tokenizer and model for the selected model
# tokenizer = AutoTokenizer.from_pretrained(models[model_name])
# model = AutoModelForSequenceClassification.from_pretrained(models[model_name])

# Create a button to apply the task to the context using the selected model
if st.button("Submit"):
    data_task = context+'\n'+task    
    settings = {'output_max_length': np.int32(max_out), 'input_max_length': np.int32(max_in), 
                'device': device, 'temprature': float(temprature)}

    generated_outputs, time_spans = ft5_models.generate(models[model_name], data_task, settings, settings['device'])

    # Tokenize the context and task
    # inputs = tokenizer(context, task, return_tensors="pt")

    # Apply the model to the inputs
    # outputs = model(**inputs)

    # Get the predicted label (assuming a binary classification task)
    # label = "Positive" if outputs.logits.argmax().item() == 1 else "Negative"

    # Show the predicted label
    st.write("Response:", generated_outputs + '\n\n Time taken: ' + str(time_spans) + ' s')

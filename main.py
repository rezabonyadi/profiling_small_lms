import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from language_models.model_runner import model_runner

import numpy as np

# Define the available models and their corresponding model names
models = {
    "Flan T5 small": "google/flan-t5-small",
    "Flan T5 base": "google/flan-t5-base",
    "Flan T5 large": "google/flan-t5-large",
    "Flan T5 xl": "google/flan-t5-xl", 
    "Flan T5 xxl": "google/flan-t5-xxl", 
    "Flan ul xl": "google/flan-ul2", 
    "GPT-2": "gpt2",   
    "GPT-2 XL": "gpt2-xl",   
    "gpt-neo": "EleutherAI/gpt-neo-1.3B",
    "alpaca": "llama"    
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

runner = model_runner()

# Create a button to apply the task to the context using the selected model
if st.button("Submit"):
    # data_task = context+'\n'+task    
    settings = {'output_max_length': np.int32(max_out), 'input_max_length': np.int32(max_in), 
                'device': device, 'temprature': float(temprature)}
    
    runner.update_model(models[model_name], settings, settings['device'])   

    generated_outputs, time_spans = runner.generate(context, task)
    st.write("Response:", generated_outputs + '\n\n Time taken: ' + str(time_spans) + ' s')

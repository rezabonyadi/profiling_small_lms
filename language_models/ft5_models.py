# Functions for running language models
from transformers import pipeline
import torch
import time

# 'google/flan-t5-small',
# 'google/flan-t5-base',
# 'google/flan-t5-large',
# 'google/flan-t5-xl',

def generate(model_name, text, settings, device='cpu'):
    input_max_length = settings['input_max_length']
    output_max_length = settings['output_max_length']
    temperature = settings['temprature']
    print(temperature)

    generator = pipeline("text2text-generation", model=model_name, truncation=True, max_length=input_max_length, device=device)
    start = time.time()
    output = generator(text, max_length=output_max_length, do_sample=True, temperature=temperature)
    end = time.time() - start    
    
    return output[0]['generated_text'], end

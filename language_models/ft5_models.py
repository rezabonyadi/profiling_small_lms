# Functions for running language models
from transformers import pipeline
import torch
import time

# 'google/flan-t5-small',
# 'google/flan-t5-base',
# 'google/flan-t5-large',
# 'google/flan-t5-xl',

class ft5_models:
    def __init__(self, model_name, settings, device='cpu'):
        self.input_max_length = settings['input_max_length']
        self.output_max_length = settings['output_max_length']
        self.temperature = settings['temprature']
        # model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)                                                                 

        self.model = pipeline("text2text-generation", model=model_name, 
                                  truncation=True, max_length=self.input_max_length, device=device)

    def generate(self, context, task):
        text = context+'\n'+task         
        start = time.time()
        output = self.model(text, max_length=self.output_max_length, do_sample=True, temperature=self.temperature)
        end = time.time() - start    
        
        return output[0]['generated_text'], end

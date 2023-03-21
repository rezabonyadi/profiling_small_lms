# Functions for running language models
from transformers import pipeline
from transformers import T5ForConditionalGeneration, AutoTokenizer

import torch
import time

# 'google/flan-t5-small',
# 'google/flan-t5-base',
# 'google/flan-t5-large',
# 'google/flan-t5-xl',

class ft5_models:
    def __init__(self, model_name, settings):
        self.input_max_length = settings['input_max_length']
        self.output_max_length = settings['output_max_length']
        self.temperature = settings['temprature']
        print(settings)
        
        if settings['is_model_large']:
            self.model = pipeline("text2text-generation", model=model_name, 
                                    truncation=True, max_length=self.input_max_length, device_map="auto",
                                    model_kwargs={"load_in_8bit": True})
        else:
            self.model = pipeline("text2text-generation", model=model_name, 
                                    truncation=True, max_length=self.input_max_length, device=settings['device'])

    def generate(self, context, task):
        text = context+'\n'+task         
        start = time.time()
        output = self.model(text, max_length=self.output_max_length, do_sample=True, temperature=self.temperature)
        end = time.time() - start    
        
        return output[0]['generated_text'], end

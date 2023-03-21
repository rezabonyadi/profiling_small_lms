from language_models import ft5_models
from language_models import gpt_models
from language_models import llama_models
import time

models_map = {
            't5': ['google/flan-t5-small', 'google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl'],
            'gpt': ['gpt2', 'gpt2-xl', 'EleutherAI/gpt-neo-1.3B'],
            'llama': ['llama']
        }

class model_runner:
    def __init__(self):
        self.model_to_run = None
        self.current_loaded_model_name = None
        self.settings = None
        self.device = None
    
    def update_model(self, model_name, settings, device):
        self.settings = settings 
        self.device = device
        print(self.current_loaded_model_name, model_name)

        if self.current_loaded_model_name != model_name:            
            if model_name in models_map['t5']:
                print('Updating model to ', model_name)
                self.current_loaded_model_name = model_name
                self.model_to_run = ft5_models.ft5_models(model_name, self.settings, self.device)

            if model_name in models_map['gpt']:
                print('Updating model to ', model_name)
                self.current_loaded_model_name = model_name
                self.model_to_run = gpt_models.gpt_models(model_name, self.settings, self.device)
            if model_name in models_map['llama']:
                print('Updating model to ', model_name)
                self.current_loaded_model_name = model_name
                self.model_to_run = llama_models.llama_models(model_name, self.settings, self.device)
            
            
            
    def generate(self, context, task):
        print(self.current_loaded_model_name)    
        if self.current_loaded_model_name is None:
            return "LOAD THE MODEL FIRST!"  
        print('Running')
        output, end = self.model_to_run.generate(context, task)
        
        return output, end
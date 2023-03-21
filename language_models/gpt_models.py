from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time

# models = {
#     "gpt-2": "gpt2-xl",
#     "gpt-neo": "EleutherAI/gpt-neo-1.3B"
# }

# model_name = "gpt-neo"
# model = AutoModelForCausalLM.from_pretrained(models[model_name])
# tokenizer = AutoTokenizer.from_pretrained(models[model_name])

class gpt_models:
    def __init__(self, model_name, settings):
        self.input_max_length = settings['input_max_length']
        self.output_max_length = settings['output_max_length']
        self.temperature = settings['temprature']
        self.device = settings['device']

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)


        # self.model = pipeline("text2text-generation", model=model_name, 
        #                           truncation=True, max_length=self.input_max_length, device=device)
        
    def generate_prompt(self, input, instruction):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            {instruction}

            ### Input:
            {input}

            ### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

            ### Instruction:
            {instruction}

            ### Response:"""

    # def generate(self, text):        
    #     start = time.time()
    #     output = self.model(text, max_length=self.output_max_length, do_sample=True, temperature=self.temperature)
    #     end = time.time() - start    
        
    #     return output[0]['generated_text'], end

    def generate(self, instruction, input):
        # instruction is the task, input is the context
        outputs = []
        start = time.time()
        input_text = self.generate_prompt(instruction, input)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        output_ids = self.model.generate(input_ids=input_ids, max_new_tokens=self.input_max_length, do_sample=False)

        span = time.time() - start

        for output_id in output_ids:
            response = self.tokenizer.decode(output_id, skip_special_tokens=True)
            clean_response = response.split("### Response:")[1].strip()
            print(response)
            outputs.append(clean_response)

        return outputs[0], span
      
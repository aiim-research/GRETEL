from google import genai
# import ollama
import os
import time
import random as rd
from src.core.llm_base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM #, BitsAndBytesConfig
import torch

from src.utils.logger import GLogger


class GeminiExplainer(LLM):

    def init(self):
        self.api_key= "AIzaSyCIOT_W5yg0s-Yan1A1StnHRftEl4OI4jk"
        self.model = "gemini-2.5-pro"
        self.client = genai.Client(api_key = self.api_key)

    def explain_counterfactual(self, prompt):

        num_tries = 0
        while num_tries < 30:
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                
                return resp.text
            except Exception as e:
                num_tries += 1
                time.sleep(60) # wait for 60 seconds before retrying
                
        raise Exception("Failed to get response from the model after multiple attempts.")
    

class LocalLlamaExplainer(LLM):

    def init(self):
        self.repo_id =  "meta-llama/Llama-3.2-1B" # "Qwen/Qwen3-0.6B" "meta-llama/Llama-3.1-8B" "meta-llama/Llama-3.2-1B"

        '''bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,   # or torch.float16
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",               # good default
        )'''

        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.repo_id, device_map="auto",trust_remote_code=True) #quantization_config=bnb_config

        # Mover a GPU si está disponible
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        if self.device == "mps":
            self.model = self.model.to(torch.float32)
        self.model = self.model.to(self.device)
        GLogger.getLogger().info('LLM - Model '+self.repo_id+' Loaded')


    def explain_counterfactual(self, prompt):      
        self.model.eval()
        # Tokenizar
 
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        
        num_tries = 0
        while num_tries < 30:
            try:
                # Generar texto
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,  # cuántos tokens generar
                        temperature=0.7,     # creatividad
                        do_sample=True,      # True para sampling
                        top_p=0.9,           # nucleus sampling
                        repetition_penalty=1.2
                    )

                # Decodificar y mostrar
                resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                GLogger.getLogger().info('Generated...')
                del inputs, outputs  # drop GPU references
                torch.cuda.empty_cache()
                return resp
            except Exception as e:
                num_tries += 1
                GLogger.getLogger().info(e)
            
        raise Exception("Failed to get response from the model after multiple attempts.")

# class OllamaExplainer(LLMExplainer):

#     def __init__(self):
#         api_key= None
#         model = "llama3"
#         super().__init__(api_key, model)

#     def explain_counterfactual(self, prompt):

#         seconds = rd.randint(1, 20)
#         # time.sleep(seconds) # wait for 60 seconds before retrying
#         client = ollama.Client()


#         num_tries = 0
#         # while num_tries < 30:
#         try:
#             resp = client.generate(
#                 model=self.model,
#                 prompt=prompt,
#                 stream=False # Configuramos stream en False para obtener la respuesta completa de una vez
#             )
            
#             return resp.response
#         except Exception as e:
#             num_tries += 1
#             # time.sleep(60) # wait for 60 seconds before retrying
            
#         # raise Exception("Failed to get response from the model after multiple attempts.")

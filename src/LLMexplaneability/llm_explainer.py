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

    def explain_counterfactual(self, system, prompt):

        num_tries = 0
        while num_tries < 30:
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents= system+prompt
                )
                
                return resp.text
            except Exception as e:
                num_tries += 1
                time.sleep(60) # wait for 60 seconds before retrying
                
        raise Exception("Failed to get response from the model after multiple attempts.")
    

class LocalLlamaExplainer(LLM):

    def init(self):
        self.repo_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)

        # Chat template (fine)
        self.tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}
<|im_start|>assistant
"""

        # ---- MODEL LOADING / DEVICE HANDLING ----
        # Drop device_map="auto" and use a single explicit device
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.repo_id,
            dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device)

        if self.device == "mps":
            # MPS does not like bfloat16/float16
            self.model = self.model.to(torch.float32)

        # Ensure pad token is set (Llama often has pad = eos)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        GLogger.getLogger().info('LLM - Model ' + self.repo_id + ' Loaded')


    def explain_counterfactual(self, system, prompt):
        self.model.eval()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        # 1) Build chat text via template (string, not tokens yet)
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 2) Tokenize normally to get input_ids + attention_mask
        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
        )

        # 3) Move EVERYTHING to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        num_tries = 0
        while num_tries < 30:
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,                      # input_ids + attention_mask
                        max_new_tokens=1024,
                        pad_token_id=self.tokenizer.pad_token_id,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.2,
                    )

                # 1) Longitud del prompt (input)
                input_length = inputs["input_ids"].shape[1]

                # 2) Nos quedamos solo con los tokens generados
                generated_tokens = outputs[0][input_length:]

                # 3) Decodificamos SOLO la respuesta del assistant
                resp = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                ).strip()

                # Optional: free GPU memory if you're looping a lot
                # del inputs, outputs
                if self.device == "cuda":
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

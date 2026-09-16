from google import genai
# import ollama
import os
import time
import random as rd
from src.core.llm_base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from src.utils.logger import GLogger


# Las claves nunca van en el codigo: viven en api_keys.txt (en la raiz del
# repo, ignorado por git). Se puede apuntar a otro fichero con la variable de
# entorno GRETEL_API_KEYS_FILE, o dar una sola clave con <PROVEEDOR>_API_KEY.
DEFAULT_API_KEYS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "api_keys.txt",
)


def load_api_keys(provider, path=None):
    """Devuelve la lista de claves declaradas para `provider`, en orden de fichero.

    Formato del fichero: una linea "PROVEEDOR=clave", '#' inicia comentario.
    Si no hay ninguna clave lanza un error, para que una ejecucion mal
    configurada falle aqui y no contra la API con una clave vacia.
    """
    provider = provider.upper()
    path = path or os.environ.get("GRETEL_API_KEYS_FILE", DEFAULT_API_KEYS_FILE)

    keys = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                if "=" in line:
                    name, _, value = line.partition("=")
                    if name.strip().upper() != provider:
                        continue
                    line = value.strip()
                if line and line not in keys:
                    keys.append(line)

    env_key = os.environ.get(provider + "_API_KEY")
    if env_key and env_key not in keys:
        keys.append(env_key)

    if not keys:
        raise RuntimeError(
            "No hay ninguna clave de " + provider + ". Anade una linea '"
            + provider + "=<clave>' en " + path
            + " (el fichero esta en .gitignore) o exporta "
            + provider + "_API_KEY."
        )
    return keys


class GeminiExplainer(LLM):

    def __init__(self):
        self.apis = load_api_keys("GEMINI")
        self.index = 0
        self.api_key = self.apis[self.index]
        self.model = "gemini-2.5-flash"
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
                self.index += 1 
                GLogger.getLogger().info("Cambiando de modelo")
                self.api_key = self.apis[self.index % len(self.apis)]
                self.client = genai.Client(api_key = self.api_key)

                time.sleep(60) # wait for 60 seconds before retrying
                
        raise Exception("Failed to get response from the model after multiple attempts.")
    

class LocalLlamaExplainer(LLM):

    def init(self):
        # self.repo_id = "meta-llama/Meta-Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-72B-Instruct"
        self.repo_id = "meta-llama/Llama-3.1-8B-Instruct"


        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,   # or torch.float16
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",               # good default
        )


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

        GLogger.getLogger().info("torch dtype = " + str(torch_dtype))

        self.model = AutoModelForCausalLM.from_pretrained(
            self.repo_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            quantization_config=bnb_config,
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
                if self.device.startswith("cuda"):
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
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

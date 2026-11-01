"""Local HuggingFace backend for the LLM contract (system, prompt) -> text.

One class for generators and judges alike; the model is a config parameter,
so every run's identity carries the exact model, quantization and sampling
settings. Two engines behind the same interface:

- "transformers": plain HF generate, one prompt at a time. Needs no extra
  dependency.
- "vllm": batched inference through vLLM, an order of magnitude faster on a
  whole dataset. Needs `pip install vllm`.

Parameters (all hashed into the component name):
- model: HF repo id, e.g. "Qwen/Qwen3-8B"
- engine: "transformers" | "vllm"
- dtype: "bfloat16" | "float16" | "auto"
- quantization: None, or for vllm "awq" | "gptq" | "fp8"; for transformers
  "4bit" | "8bit" (bitsandbytes). MXFP4 checkpoints (gpt-oss) load as-is.
- temperature, top_p, seed, max_new_tokens: sampling controls
- enable_thinking: chat-template switch for hybrid models (Qwen3); False
  keeps the output to the answer only
- reasoning_effort: chat-template switch for gpt-oss ("low"|"medium"|"high")
- tensor_parallel_size, gpu_memory_utilization, max_model_len: vllm only
- max_num_seqs, limit_mm_per_prompt: optional vllm concurrency/modality limits
- revision: checkpoint commit or ref; local_files_only uses its cached snapshot
- system_prefix: model-specific system instruction, e.g. Muse reasoning strength
- skip_special_tokens: vllm decoding; defaults to False for Muse channel framing

Weights are cached under HF_HOME; point it at a large disk.
"""
import re
import os
from pathlib import Path

from src.core.llm_base import LLM
from src.utils.logger import GLogger

# Hybrid models emit an (often empty) think block before the answer
_THINK_BLOCK = re.compile(r'<think>.*?</think>\s*', re.DOTALL)


class _LLMAnswer(str):
    """Keep the text API while carrying decoding diagnostics to probe records."""

    def __new__(cls, text, raw, finish_reason=None, stop_reason=None, token_count=None):
        answer = super().__new__(cls, text)
        answer.llm_metadata = {'judge_raw_output': raw, 'finish_reason': finish_reason,
                               'stop_reason': stop_reason, 'output_token_count': token_count}
        return answer

    def __reduce_ex__(self, protocol):
        m = self.llm_metadata
        return type(self), (str(self), m['judge_raw_output'], m['finish_reason'],
                           m['stop_reason'], m['output_token_count'])


class HuggingFaceLLM(LLM):

    def check_configuration(self):
        super().check_configuration()
        p = self.local_config['parameters']
        if 'model' not in p:
            raise ValueError('HuggingFaceLLM needs a "model" parameter (HF repo id)')
        p.setdefault('engine', 'transformers')
        if p['engine'] not in ('transformers', 'vllm'):
            raise ValueError('HuggingFaceLLM "engine" must be "transformers" or "vllm"')
        p.setdefault('dtype', 'bfloat16')
        p.setdefault('quantization', None)
        p.setdefault('temperature', 0.0)
        p.setdefault('top_p', 1.0)
        p.setdefault('seed', 0)
        p.setdefault('max_new_tokens', 1024)
        p.setdefault('enable_thinking', False)
        p.setdefault('reasoning_effort', None)
        p.setdefault('tensor_parallel_size', 1)
        p.setdefault('gpu_memory_utilization', 0.9)
        p.setdefault('max_model_len', None)
        p.setdefault('revision', None)
        p.setdefault('local_files_only', False)
        p.setdefault('system_prefix', '')
        p.setdefault('max_num_seqs', None)
        p.setdefault('limit_mm_per_prompt', None)
        p.setdefault('skip_special_tokens', 'muse-glimmer' not in p['model'].lower())
        if type(p['local_files_only']) is not bool or not isinstance(p['system_prefix'], str):
            raise ValueError('local_files_only must be boolean and system_prefix must be text')
        if type(p['skip_special_tokens']) is not bool:
            raise ValueError('skip_special_tokens must be boolean')

    def init(self):
        super().init()
        p = self.local_config['parameters']
        self.model_id = p['model']
        self.engine = p['engine']
        self.temperature = p['temperature']
        self.top_p = p['top_p']
        self.seed = p['seed']
        self.system_prefix = p['system_prefix']
        self.max_new_tokens = p['max_new_tokens']
        # Extra variables handed to the chat template; templates that do not
        # know a variable simply ignore it
        self.chat_kwargs = {'enable_thinking': bool(p['enable_thinking'])}
        if p['reasoning_effort'] is not None:
            self.chat_kwargs['reasoning_effort'] = p['reasoning_effort']

        if self.engine == 'vllm':
            self._init_vllm(p)
        else:
            self._init_transformers(p)
        GLogger.getLogger().info(f'LLM - {self.model_id} loaded with {self.engine}')

    # ------------------------------------------------------------------

    def _init_vllm(self, p):
        # GRETEL may initialize CUDA while preparing the oracle before this model.
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        from vllm import LLM as VllmEngine, SamplingParams
        model_path = self._model_path(p)
        extra = {}
        for key in ('max_num_seqs', 'limit_mm_per_prompt'):
            if p[key] is not None:
                extra[key] = p[key]
        if p['revision'] is not None and not p['local_files_only']:
            extra['revision'] = p['revision']
        self.llm = VllmEngine(model=model_path,
                              dtype=p['dtype'],
                              quantization=p['quantization'],
                              tensor_parallel_size=p['tensor_parallel_size'],
                              gpu_memory_utilization=p['gpu_memory_utilization'],
                              max_model_len=p['max_model_len'],
                              seed=self.seed, **extra)
        self.sampling = SamplingParams(temperature=self.temperature, top_p=self.top_p,
                                       max_tokens=self.max_new_tokens, seed=self.seed,
                                       skip_special_tokens=p.get('skip_special_tokens',
                                           'muse-glimmer' not in self.model_id.lower()),
                                       spaces_between_special_tokens=False)

    def _init_transformers(self, p):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        model_path = self._model_path(p)
        load_kwargs = {'local_files_only': p['local_files_only']}
        if p['revision'] is not None and not p['local_files_only']:
            load_kwargs['revision'] = p['revision']
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {'device_map': 'auto'}
        if p['quantization'] in ('4bit', '8bit'):
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=p['quantization'] == '4bit',
                load_in_8bit=p['quantization'] == '8bit',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type='nf4')
        elif p['dtype'] != 'auto':
            kwargs['dtype'] = getattr(torch, p['dtype'])
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs, **load_kwargs)
        self.model.eval()

    # ------------------------------------------------------------------

    def _model_path(self, p):
        if not p['local_files_only']:
            return self.model_id
        from huggingface_hub import try_to_load_from_cache
        config = try_to_load_from_cache(self.model_id, 'config.json', revision=p['revision'])
        if not isinstance(config, str) or not Path(config).is_file():
            raise FileNotFoundError(f'No cached config.json for {self.model_id} at {p["revision"] or "main"}')
        # Use the standard inference snapshot directly; original/metal exports
        # need not be downloaded. The engine still validates its actual weights.
        snapshot = Path(config).parent
        p['resolved_revision'] = snapshot.name
        return str(snapshot)

    def _messages(self, system, prompt):
        prefix = getattr(self, 'system_prefix', '')
        if prefix:
            system = prefix.rstrip() + '\n' + system
        return [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]

    @staticmethod
    def _clean(text):
        if '<think>' in (text or '') and '</think>' not in text:
            return ''  # Incomplete reasoning is not a final answer.
        text = _THINK_BLOCK.sub('', text or '').strip()
        if '</think>' in text:
            # The opening think token may be part of the generation prompt.
            text = text.rsplit('</think>', 1)[1].strip()
        if text.startswith(('to=self', 'to=user', 'assistant to=self',
                            'assistant to=user', '<|start|>assistant')):
            user_messages = list(re.finditer(r'(?:assistant )?to=user<\|message\|>', text))
            if user_messages:
                # The assistant role can already be part of the input prompt.
                text = text[user_messages[-1].end():]
            elif text.startswith(('to=self', 'assistant to=self', '<|start|>assistant to=self')):
                return ''
        # Keep the final channel when gpt-oss returns Harmony channel markers.
        if '<|channel|>final' in text:
            text = text.rsplit('<|channel|>final', 1)[1]
            text = text.removeprefix('<|message|>')
        elif text.startswith(('analysis', 'assistantanalysis', 'assistantfinal')) and 'assistantfinal' in text:
            # Some vLLM outputs start at the analysis channel, without the role.
            text = text.rsplit('assistantfinal', 1)[1]
        elif '<|channel|>analysis' in text or '<|channel|>commentary' in text:
            return ''
        elif re.match(r'^(?:assistant)?analysis(?=\S)', text):
            return ''  # A flattened analysis channel without a final answer.
        for marker in ('<|fim_suffix|>', '<|im_end|>', '<|return|>', '<|end|>', '<|eom|>'):
            text = text.replace(marker, '')
        return text.strip()

    def explain_counterfactual(self, system, prompt):
        return self.explain_many([(system, prompt)])[0]

    def explain_many(self, pairs):
        """Answer a list of (system, prompt) pairs; batched under vllm."""
        messages = [self._messages(s, p) for s, p in pairs]
        if self.engine == 'vllm':
            outputs = self.llm.chat(messages, self.sampling, use_tqdm=False,
                                    chat_template_kwargs=self.chat_kwargs)
            answers = []
            for output in outputs:
                completion = output.outputs[0]
                raw = completion.text
                tokens = getattr(completion, 'token_ids', None)
                answers.append(_LLMAnswer(self._clean(raw), raw,
                    getattr(completion, 'finish_reason', None),
                    getattr(completion, 'stop_reason', None),
                    len(tokens) if tokens is not None else None))
            return answers
        return [self._generate_one(m) for m in messages]

    def _generate_one(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, **self.chat_kwargs)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        self.torch.manual_seed(self.seed)
        generate_kwargs = {'max_new_tokens': self.max_new_tokens,
                           'pad_token_id': self.tokenizer.pad_token_id,
                           'do_sample': self.temperature > 0}
        if self.temperature > 0:
            generate_kwargs.update(temperature=self.temperature, top_p=self.top_p)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, **generate_kwargs)
        generated = output[0][inputs['input_ids'].shape[1]:]
        # Preserve channel markers so analysis and final remain distinguishable.
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        return _LLMAnswer(self._clean(raw), raw, token_count=len(generated))

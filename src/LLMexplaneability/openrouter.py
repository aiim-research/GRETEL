"""OpenRouter backend for the LLM contract (system, prompt) -> text.

OpenRouter exposes an OpenAI-compatible endpoint, so the same class serves any
model it hosts, for generators and judges alike. Kept in its own module so a
judge-only run does not import the local HuggingFace stack.

Credentials come from the environment: export OPENROUTER_API_KEY.

Parameters (all hashed into the component name):
- model: OpenRouter model id, e.g. "deepseek/deepseek-v3.2"
- temperature, seed, max_tokens: sampling controls (seed is honoured by the
  providers that support it)
- provider: optional provider slug to pin (e.g. "DeepSeek"); with it set,
  fallbacks to other providers are disabled so the served weights and
  quantization stay fixed across runs
- retries, retry_sleep: transient-error handling
"""
import os
import time

from src.core.llm_base import LLM
from src.utils.logger import GLogger


class OpenRouterLLM(LLM):

    def check_configuration(self):
        super().check_configuration()
        p = self.local_config['parameters']
        if 'model' not in p:
            raise ValueError('OpenRouterLLM needs a "model" parameter (OpenRouter model id)')
        p.setdefault('temperature', 0.0)
        p.setdefault('seed', 0)
        p.setdefault('max_tokens', 2048)
        p.setdefault('provider', None)
        p.setdefault('retries', 3)
        p.setdefault('retry_sleep', 10)

    def init(self):
        super().init()
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise SystemExit('OpenRouterLLM needs an API key: export OPENROUTER_API_KEY before running.')

        from openai import OpenAI
        self.client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=api_key)

        p = self.local_config['parameters']
        self.model = p['model']
        self.temperature = p['temperature']
        self.seed = p['seed']
        self.max_tokens = p['max_tokens']
        self.provider = p['provider']
        self.retries = p['retries']
        self.retry_sleep = p['retry_sleep']

    def explain_counterfactual(self, system, prompt):
        extra_body = {}
        if self.provider:
            extra_body['provider'] = {'order': [self.provider], 'allow_fallbacks': False}

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'system', 'content': system},
                              {'role': 'user', 'content': prompt}],
                    temperature=self.temperature,
                    seed=self.seed,
                    max_tokens=self.max_tokens,
                    extra_body=extra_body or None,
                )
                return completion.choices[0].message.content
            except Exception as e:
                last_error = e
                GLogger.getLogger().info(
                    f'OpenRouter call to {self.model} failed ({type(e).__name__}), '
                    f'attempt {attempt + 1}/{self.retries + 1}')
                time.sleep(self.retry_sleep)

        raise RuntimeError(f'OpenRouter call to {self.model} failed after {self.retries + 1} attempts') from last_error

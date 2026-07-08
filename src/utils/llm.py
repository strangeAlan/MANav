import base64
import os
from openai import OpenAI
from io import BytesIO
import requests


def _image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def _is_local_qwen(model):
    return str(model).strip().lower() in {'local_qwen', 'qwen', 'transformers_qwen'}


class LocalQwenHTTP:
    def __init__(self, base_url=None, timeout=180):
        self.base_url = (
            base_url
            or os.environ.get('UNIGOAL_QWEN_SERVER_URL')
            or os.environ.get('MVPNAV_QWEN_SERVER_URL')
            or 'http://127.0.0.1:18080'
        ).rstrip('/')
        self.timeout = timeout

    def generate(self, prompt, images=None):
        payload = {
            'prompt': prompt,
            'images': images or [],
        }
        response = requests.post(
            f'{self.base_url}/generate',
            json=payload,
            timeout=self.timeout,
            proxies={'http': '', 'https': ''},
        )
        response.raise_for_status()
        return response.json().get('text', '')


class LLM:
    def __init__(self, base_url, api_key, llm_model):
        self.base_url = base_url
        self.api_key = api_key
        self.llm_model = llm_model
        self.local_qwen = LocalQwenHTTP(base_url) if _is_local_qwen(llm_model) else None

    def __call__(self, prompt):
        if self.local_qwen is not None:
            return self.local_qwen.generate(prompt, images=[])

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            model=self.llm_model,
        )
        return chat_completion.choices[0].message.content


class VLM:
    def __init__(self, base_url, api_key, vlm_model):
        self.base_url = base_url
        self.api_key = api_key
        self.vlm_model = vlm_model
        self.local_qwen = LocalQwenHTTP(base_url) if _is_local_qwen(vlm_model) else None

    def __call__(self, prompt, image):
        image_str = _image_to_base64(image)

        if self.local_qwen is not None:
            return self.local_qwen.generate(prompt, images=[image_str])

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": "data:image/png;base64," + image_str}
                    ]
                }
            ],
            model=self.vlm_model,
        )
        return chat_completion.choices[0].message.content

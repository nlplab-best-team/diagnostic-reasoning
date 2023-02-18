import os
import openai
from argparse import Namespace

class ModelAPI(object):
    cost_per_1000tokens = 0.02

    def __init__(self, config: Namespace):
        self._config = config
        self._openai = openai
        self._openai.api_key = os.getenv("OPENAI_API_KEY")

    def complete(self, prompt: str) -> dict:
        return self._openai.Completion.create(prompt=prompt, **vars(self._config))
    
    def set_api_key(self, key: str) -> None:
        self._openai.api_key = key

if __name__ == "__main__":
    # unit testing
    config = Namespace(
        model="text-davinci-003",
        max_tokens=7,
        temperature=0
    )

    model_api = ModelAPI(config)

    res = model_api.complete(prompt="Say this is a test")
    print(res)
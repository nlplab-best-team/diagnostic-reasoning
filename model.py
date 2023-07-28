import os
import time
import openai
from colorama import Fore, Style
from argparse import Namespace

class ModelAPI(object):
    cost_per_1000tokens = 0.02

    def __init__(self, config: Namespace):
        self._config = config
        self._openai = openai
        self._openai.api_key = os.getenv("OPENAI_API_KEY")
        
    def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        max_retries: int = 5,
        errors: tuple = (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError),
    ):
        """Retry a function with exponential backoff."""
    
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay
            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)
                # Retry on specific errors
                except errors as e:
                    # Increment retries
                    num_retries += 1
                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            Fore.RED + f"Maximum number of retries ({max_retries}) exceeded." + Style.RESET_ALL
                        )
                    # Increment the delay
                    delay *= exponential_base
                    # Sleep for the delay
                    print(Fore.YELLOW + f"Error encountered. Retry ({num_retries}) after {delay} seconds..." + Style.RESET_ALL)
                    time.sleep(delay)
                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e
    
        return wrapper

    @retry_with_exponential_backoff
    def complete(self, prompt: str) -> dict:
        return self._openai.Completion.create(prompt=prompt, **vars(self._config))
    
    @retry_with_exponential_backoff
    def chat(self, messages: list) -> dict:
        return self._openai.ChatCompletion.create(messages=messages, **vars(self._config))
    
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
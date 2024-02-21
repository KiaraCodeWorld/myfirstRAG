
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import requests
from dotenv import load_dotenv
import os

class CustomLLM(LLM):
    n: int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        load_dotenv()
        API_KEY = os.getenv("API_TOKEN")
        headers = {"Authorization": f"Bearer {API_KEY}"}
        API_URL = os.getenv("API_URL") #"https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud/"
        payload = {
            "inputs": prompt,
            "parameters": {  # Try and experiment with the parameters
                "max_new_tokens": 1024,
                "temperature": 0.6,
                "top_p": 0.9,
                "do_sample": False,
                "return_full_text": False
            }
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()[0]['generated_text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}

"""
print("hello")
llm = CustomLLM(n=200)
prompt = "what is the capital of france ?"
output = llm._call(prompt)
print(output)
"""
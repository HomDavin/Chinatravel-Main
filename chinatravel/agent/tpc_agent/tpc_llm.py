import os
import sys
from typing import Optional
project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root_path not in sys.path:
    sys.path.append(project_root_path)
if os.path.dirname(project_root_path) not in sys.path:
    sys.path.append(os.path.dirname(project_root_path))


from agent.llms import AbstractLLM, Qwen


class TPCLLM(AbstractLLM):


    def __init__(self, model_name: str = "Qwen3-4B", max_model_len: Optional[int] = None):
        super().__init__()
        self._backend = Qwen(model_name, max_model_len=max_model_len)
        self.name = f"{self._backend.name}-local"

    def _get_response(self, messages, one_line, json_mode):
        response = self._backend(messages, one_line=one_line, json_mode=json_mode)
        # Mirror token statistics for downstream logging.
        self.input_token_count = self._backend.input_token_count
        self.output_token_count = self._backend.output_token_count
        self.input_token_maxx = self._backend.input_token_maxx
        return response

import json
import os
import sys
from typing import Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv("../.env")

# Make sure local tools.py is importable whether run from this folder or elsewhere
try:
    from tools import ToolBox
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from tools import ToolBox


class LlmAgent:
    _history: list[dict[str, str]]
    _system_prompt: str
    _model: str
    _debug: bool = False

    def __init__(self, system_prompt: str, model: str = 'gpt-5-nano') -> None:
        self._system_prompt = system_prompt
        self._model = model
        self.reset_history()

    def set_debug(self, debug: bool) -> None:
        self._debug = debug

    def reset_history(self) -> None:
        self._history = [
            {'role': 'system', 'content': self._system_prompt}
        ]

    async def execute(self, prompt: str, toolbox: Optional[ToolBox] = None) -> str:
        client = AsyncOpenAI()
        had_tool_calls = True
        self._history.append({
            'role': 'user', 'content': prompt
        })
        result_content = ""

        while had_tool_calls:
            response = await client.responses.create(
                input=self._history,
                model=self._model,
                tools=toolbox.tools if toolbox else []
            )

            self._history += response.output

            had_tool_calls = any(item.type == 'function_call' for item in response.output)

            for item in response.output:
                if item.type == "function_call":
                    if self._debug:
                        print(f'>>> Calling {item.name} with args {item.arguments}')
                    if func := toolbox.get_tool_function(item.name):
                        result = func(**json.loads(item.arguments))

                        if self._debug:
                            print(f'>>> {item.name} returned {result}')
                        self._history.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps(result)
                        })

            if self._debug:
                print('AI:', response.output_text)
            result_content = response.output_text

        return result_content
    
from .llm_agent import LlmAgent
from .remote_toolbox import remote_toolbox

SYSTEM_PROMPT = """
- You are a helpful AI assistant that can control a TV remote.
- You have access to tools that can control the TV remote. 
- You should use the tools to control the TV remote.

"""

class RemoteAgent(LlmAgent):
    def __init__(self):
        super().__init__(system_prompt=SYSTEM_PROMPT, model="gpt-4o-mini")

    async def evaluate(self, user_input):
        return await self.execute(
            prompt="Here is the user_input: " + user_input,
            toolbox=remote_toolbox
        )

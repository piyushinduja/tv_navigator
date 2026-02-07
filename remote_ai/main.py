import asyncio
from .remote_agent import RemoteAgent

async def ai_agent(input_text):
    print("Input to AI agent:", input_text)
    agent = RemoteAgent()
    await agent.evaluate(input_text)


if __name__ == "__main__":
    asyncio.run(ai_agent("Play the video, go back 30 seconds, and turn the volume up a lot"))
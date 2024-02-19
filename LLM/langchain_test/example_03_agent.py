from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain.llms.openai import OpenAI
from langchain.serpapi import SerpAPIWrapper
import os

os.environ[
    "SERPAPI_API_KEY"
] = "40f145cc840ac92203b852d793a3e0c2bc45fbcaec03f9c86963db222e6b1e7b"
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
        return_direct=True,
    )
]
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish


class FakeAgent(BaseSingleActionAgent):
    @property
    def input_keys(self):
        return ["input"]

    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")


agent = FakeAgent()
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)
agent_executor.run("How many people live in canada as of 2023?")

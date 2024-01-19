from langchain.agents import load_tools
from langchain.agents import initialize_agent, AgentType
from langchain.llms.openai import OpenAI
import os
import openai

openai.api_key = "sk-zk2ce44e53ac2484f5add7c2799e7d60652659fb0aa6ef8f"
openai.base_url = "https://flag.smarttrot.com/v1/"
# openai_key = "sk-zk2ce44e53ac2484f5add7c2799e7d60652659fb0aa6ef8f"
# os.environ["OPENAI_API_KEY"] = openai_key
os.environ["OPENAI_API_BASE"] = openai.api_key
os.environ["OPENAI_API_KEY"] = openai.base_url
os.environ[
    "SERPAPI_API_KEY"
] = "40f145cc840ac92203b852d793a3e0c2bc45fbcaec03f9c86963db222e6b1e7b"

llm = OpenAI(openai_api_key=openai.api_key, openai_api_base=openai.base_url)

tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent.run(
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
)

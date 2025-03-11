import getpass
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain import hub
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.agents import tool

load_dotenv()

if "AZURE_OPENAI_API_KEY" not in os.environ:
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass(
        "Enter your AzureOpenAI API key: "
    )
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
azure_openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]

# Initialize the Azure OpenAI model
llm = AzureChatOpenAI(
    azure_deployment=azure_openai_deployment_name,
    api_version=azure_openai_api_version,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize the DuckDuckGo search tool
search_tool = DuckDuckGoSearchRun()

# Initialize Open Weather Map tool
weather = OpenWeatherMapAPIWrapper()

weather_tool = load_tools(["openweathermap-api"], llm)[0]

# Define custom word counter tool
@tool
def word_counter_tool(text: str) -> int:
  """Returns the word count."""
  return len(text.split())

# Load prompt template from LangChain Hub
prompt = hub.pull("hwchase17/react")

# List of tools for the agent
tools = [search_tool, weather_tool, word_counter_tool]

# Create the agent using the LLM and the prompt template
agent = create_react_agent(tools=tools, llm=llm, prompt=prompt)

# Initialize the agent executor with the created agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

while True:
    response = agent_executor.invoke({"input": input("Enter your question: ")})

    print(response['output'])

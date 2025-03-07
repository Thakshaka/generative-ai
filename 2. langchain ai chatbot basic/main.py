import getpass
import os
from langchain_openai import AzureChatOpenAI

if "AZURE_OPENAI_API_KEY" not in os.environ:
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass(
        "Enter your AzureOpenAI API key: "
    )
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
azure_openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]

llm = AzureChatOpenAI(
    azure_deployment=azure_openai_deployment_name,
    api_version=azure_openai_api_version,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)

print(ai_msg.content)

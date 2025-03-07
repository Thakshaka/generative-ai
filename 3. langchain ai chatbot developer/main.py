import getpass
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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

parser = StrOutputParser() # Output parser to convert the response to a string

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an intelligent assistant. Answer the following question."),
    MessagesPlaceholder(variable_name="history"),
    MessagesPlaceholder(variable_name="question")
])

history = [
    HumanMessage(content="My name is Thakshaka"),
    AIMessage(content="Nice to meet you, Thakshaka!"),
]

chain = prompt | llm | parser

while True:
    question = input("Enter your question: ")

    ai_msg = chain.invoke(
        {
            "history": history,
            "question": [HumanMessage(content=question)]
        }
    )

    print(ai_msg)
    history.extend([HumanMessage(content=question), AIMessage(content=ai_msg)])

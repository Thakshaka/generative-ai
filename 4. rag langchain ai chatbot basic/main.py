import getpass
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

if "AZURE_OPENAI_API_KEY" not in os.environ:
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass(
        "Enter your AzureOpenAI API key: "
    )
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
azure_openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]
embedding_model = os.environ["EMBEDDING_MODEL"]

llm = AzureChatOpenAI(
    azure_deployment=azure_openai_deployment_name,
    api_version=azure_openai_api_version,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embeddings = AzureOpenAIEmbeddings(
    model=embedding_model
)

documents = [
    Document(
        page_content="WSO2’s API management provides the #1 open source, market-leading full lifecycle platform for building, integrating, securing, and exposing AI and digital services as managed APIs in the cloud, on-premises, hybrid architectures, and modern environments like Kubernetes.",
        metadata={"source": "WSO2 API Management"},
    ),
    Document(
        page_content="WSO2’s API management provides the #1 open source, market-leading full lifecycle platform for building, integrating, securing, and exposing AI and digital services as managed APIs in the cloud, on-premises, hybrid architectures, and modern environments like Kubernetes.Power continuous innovation and create unique digital experiences by connecting anything to anything with the AI-powered WSO2 Integration Platform. Break down data silos, boost productivity, and streamline workflows. Develop integrations in low-code and pro-code, and deploy them anywhere in a microservice or ESB-style architecture.",
        metadata={"source": "WSO2 Integration Platform"},
    ),
    Document(
        page_content="Exceptional digital experiences demand both secure and convenient access to resources, whether for employees, consumers, business customers or APIs. WSO2 offers the flexible, extensible identity and access management (IAM) products you need, offered in your choice of multi-tenant SaaS, single-tenant private SaaS, or open source software.",
        metadata={"source": "WSO2 Identity and Access Management"},
    ),
    Document(
        page_content="Choreo goes beyond the infrastructure automation capabilities of a typical internal developer platform. Its self-serviceable capabilities free up developers to be more creative and productive, helping businesses deliver applications faster.",
        metadata={"source": "WSO2 Choreo"}
    ),
    Document(
        page_content="Today's breakfast menu at WSO2. Supplier: Healthy Cafe - Garlic Kandha, Hathawariya Kandha. Supplier: Spirit Kitchen - Kawpi with lunumiris and grated coconut",
        metadata={"source": "Today's breakfast menu at WSO2"},
    ),
]

vector_store = Chroma(embedding_function=embeddings)

vector_store = Chroma.from_documents(
    documents,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

response = chain.invoke("What is today's lunch menu at WSO2?")

print(response.content)

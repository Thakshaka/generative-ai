import getpass
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

if "AZURE_OPENAI_API_KEY" not in os.environ:
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass(
        "Enter your AzureOpenAI API key: "
    )
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
azure_openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]
embedding_model = os.environ["EMBEDDING_MODEL"]

# Initialize the Azure OpenAI model
llm = AzureChatOpenAI(
    azure_deployment=azure_openai_deployment_name,
    api_version=azure_openai_api_version,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize the Azure OpenAI embedding model
embeddings = AzureOpenAIEmbeddings(
    model=embedding_model
)

# Load the document
loader = TextLoader("../5. rag langchain ai chatbot developer/document.txt")
docs = loader.load()
# print(docs[0].page_content)

# Spilt the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
# print(splits)

# Create a vector store
vector_store = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
)

# Create a retriever
retriever = vector_store.as_retriever()

# # Define system prompt
# system_prompt = (
#     "You are an intelligent chatbot. Use the following context to answer the question. If you don't know the answer, or if the answer is not in the context, just say that you're here to help with questions related to IAM."
#     "\n\n"
#     "{context}"
# )

# # Create the prompt template
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}")
#     ]
# )

# # Create the question-answering chain
# qa_chain = create_stuff_documents_chain(llm, prompt)

# # Create the RAG chain
# rag_chain = create_retrieval_chain(retriever, qa_chain)

# while True:
#     # Invoke the RAG chain with example questions
#     response = rag_chain.invoke({"input": input("Ask a question: ")})

#     print(f'Answer: {response["answer"]}')

# Define the contextualize system prompt
contextualize_system_prompt = (
    "Using chat history and the latest user question, just reformulate question if needed and otherwise return it as is"
)

# Create the contextualize prompt template
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create the history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

# Define system prompt
system_prompt = (
    "You are an intelligent chatbot. Use the following context to answer the question. If you don't know the answer, or if the answer is not in the context, just say that you're here to help with questions related to IAM."
    "\n\n"
    "{context}"
)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create the question-answering chain
qa_chain = create_stuff_documents_chain(llm, prompt)

# Create the history aware RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Initialize the store for session histories
store = {}

# Function to get the session history for a given session ID
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Create the conversational RAG chain with session history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

while True:
    # Invoke the conversational RAG chain with example questions
    response = conversational_rag_chain.invoke(
        {"input": input("Ask a question: ")},
        config={"configurable": {"session_id": "101"}},
    )

    print(response["answer"])

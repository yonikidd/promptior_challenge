from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes
from fastapi import FastAPI
from dotenv import load_dotenv
from typing import List

load_dotenv()
app = FastAPI(
        title="Chatbot Promptior",
        version="1.0",
        description="A simple API server using LangChain's Runnable interfaces",
    )

class Input(BaseModel):
        input: str
        chat_history: List[BaseMessage] = Field(
            ...,
            extra={"widget": {"type": "chat", "input": "location"}},
        )

class Output(BaseModel):
    output: str

def load_data(directory):
    loader = DirectoryLoader(directory, glob='*.md')
    return loader.load()

def split_documents(data):
    text_splitter = RecursiveCharacterTextSplitter()
    return text_splitter.split_documents(data)

def create_embeddings(documents):
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    return vector

def create_retriever(embeddings, name, description):
    retriever = embeddings.as_retriever()
    retriever_tool = create_retriever_tool(
         retriever,
         name,
         description
    )
    tools = [retriever_tool]
    return tools

def create_agent_executor():
    prompt = hub.pull("hwchase17/openai-functions-agent")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor




if __name__ == '__main__':

    DATA_DIR = "data/promptior"
    data = load_data(DATA_DIR)

    documents = split_documents(data)

    embeddings = create_embeddings(documents)
    tools = create_retriever(
        embeddings,
        "promptior_search",
        "Search for information about Promptior. For any questions about Promptior, you must use this tool!",
    )

    agent_executor = create_agent_executor()

    add_routes(app, agent_executor.with_types(input_type=Input, output_type=Output), path="/agent")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

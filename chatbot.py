from langchain.document_loaders import DirectoryLoader
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

DATA_DIR = 'data/promptior'

def load_data(directory):
    loader = DirectoryLoader(directory, glob='*.md')
    return loader.load()

def split_documents(data):
    text_splitter = RecursiveCharacterTextSplitter()
    return text_splitter.split_documents(data)

def create_retriever(documents, embeddings):
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    return create_retriever_tool(retriever, 
                                 "promptior_search", 
                                 "Search for information about Promptior. For any questions about Promptior, you must use this tool!")

def create_agent(llm, tools, prompt):
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, 
                         tools=tools, 
                         verbose=True)

def create_app(agent_executor, title, version, description, path):
    app = FastAPI(
        title=title,
        version=version,
        description=description,
    )

    class Input(BaseModel):
        input: str
        chat_history: List[BaseMessage] = Field(
            ...,
            extra={"widget": {"type": "chat", "input": "location"}},
        )

    class Output(BaseModel):
        output: str

    add_routes(app, agent_executor.with_types(input_type=Input, output_type=Output), path=path)
    return app

if __name__ == '__main__':
    DATA_DIR = 'data/promptior'
    
    data = load_data(DATA_DIR)
    documents = split_documents(data)

    embeddings = OpenAIEmbeddings()
    retriever_tool = create_retriever(documents, embeddings)
    tools = [retriever_tool]

    prompt = hub.pull("hwchase17/openai-functions-agent")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = create_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    app = create_app(agent_executor, "Chatbot Promptior", "1.0", "A simple API server using LangChain's Runnable interfaces", "/agent")

    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

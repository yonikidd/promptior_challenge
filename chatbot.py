from langchain.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes
from typing import List


from dotenv import load_dotenv



load_dotenv()

DATA_DIR = 'data/promptior'

loader = DirectoryLoader(DATA_DIR, glob='*.md')
data = loader.load()

#no hacemos text splitter son textos chiquitos
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(data)
print(len(documents))

embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "promptior_search",
    "Search for information about Promptior. For any questions about Promptior, you must use this tool!",
)
tools = [retriever_tool]




prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent,tools = tools, verbose=True)
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

add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
from langchain.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate




from dotenv import load_dotenv
from langchain_openai import ChatOpenAI



load_dotenv()

DATA_DIR = 'data/promptior'
PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

def load_data():
    loader = DirectoryLoader(DATA_DIR, glob='*.md')
    return loader.load()

#no hacemos text splitter son textos chiquitos

def store_data_in_db(data):
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(data, embeddings)
    return vector

def asking_question(vector, input_question):
    llm = ChatOpenAI()
    document_chain = create_stuff_documents_chain(llm, PROMPT_TEMPLATE)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": input_question})
    print(response["answer"])




def main():
    data = load_data()
    vector = store_data_in_db(data)
    input_question = "en qué año se fundó promptior?"
    asking_question(vector, input_question)


if __name__ == '__main__':
    main()
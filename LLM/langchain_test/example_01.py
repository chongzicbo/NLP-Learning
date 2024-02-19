import os
import openai
from langchain.llms.openai import OpenAI

openai.api_key = "sk-zk2ce44e53ac2484f5add7c2799e7d60652659fb0aa6ef8f"
openai.base_url = "https://flag.smarttrot.com/v1/"
# openai_key = "sk-zk2ce44e53ac2484f5add7c2799e7d60652659fb0aa6ef8f"
# os.environ["OPENAI_API_KEY"] = openai_key
os.environ["OPENAI_API_BASE"] = openai.api_key
os.environ["OPENAI_API_KEY"] = openai.base_url

# 1.llm chain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(openai_api_key=openai.api_key, openai_api_base=openai.base_url)

# output = llm.invoke("how to extract images from a docx ?")
# print(output)

from langchain_core.prompts import ChatPromptTemplate

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are world class technical documentation writer."),
#         ("user", "{input}"),
#     ]
# )
# chain = prompt | llm
# chain.invoke({"input": "how can langsmith help with testing?"})

# from langchain_core.output_parsers import StrOutputParser

# output_parser = StrOutputParser()

# chain = prompt | llm | output_parser
# output = chain.invoke({"input": "how can langsmith help with testing?"})
# print(output)

# 2.retrieval chain

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
print(docs)

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
document_chain.invoke(
    {
        "input": "how can langsmith help with testing?",
        "context": [
            Document(page_content="langsmith can let you visualize test results")
        ],
    }
)
from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

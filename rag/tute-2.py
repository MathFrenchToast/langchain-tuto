# local LMM RAG 
# based on  https://python.langchain.com/docs/use_cases/question_answering/

# Load documents

from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

# Split documents

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
splits = text_splitter.split_documents(loader.load())

# Embed and store splits

from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import GPT4AllEmbeddings
embeddings = GPT4AllEmbeddings(model="/dev/llm/ggml-model-gpt4all-falcon-q4_0.bin", n_ctx=512, n_threads=8)

# vectorstore = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings())
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Prompt 
# https://smith.langchain.com/hub/rlm/rag-prompt

from langchain import hub
rag_prompt = hub.pull("rlm/rag-prompt")

from langchain.llms import GPT4All
llm = GPT4All(model="/dev/llm/ggml-model-gpt4all-falcon-q4_0.bin")

# RAG chain 

from langchain.schema.runnable import RunnablePassthrough
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | rag_prompt 
    | llm 
)

rag_chain.invoke("What is Task Decomposition?")
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
from langchain.embeddings import GPT4AllEmbeddings
embeddings = GPT4AllEmbeddings(model="/dev/llm/ggml-all-MiniLM-L6-v2-f16.bin", n_ctx=512, n_threads=8)

# vectorstore = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings())
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Prompt 
# https://smith.langchain.com/hub/rlm/rag-prompt


from langchain.llms import GPT4All
# Callbacks support token-wise streaming
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model="/dev/llm/ggml-model-gpt4all-falcon-q4_0.bin", callbacks=callbacks, verbose=True)

from langchain import hub
rag_prompt = hub.pull("rlm/rag-prompt")

from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)
# RAG chain 

from langchain.schema.runnable import RunnablePassthrough
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | rag_prompt_custom 
    | llm 

)

result = rag_chain.invoke("What is Task Decomposition?")


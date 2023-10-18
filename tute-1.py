from langchain.llms import GPT4All
llm = GPT4All(model="/dev/llm/ggml-model-gpt4all-falcon-q4_0.bin")
text = "What would be a good company name for a company that makes very long socks?"

answ = llm.predict(text)
print(answ)
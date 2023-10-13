# LLM journey

with an emphasis on locally ran LLM

My setup is on windows:
- create a root c:\dev directory
- create a c:\dev\llm folder
- download falcon4 ggml model and save it to llm folder, see 'Model Explorer' section on: https://gpt4all.io/index.html
- open a powershell
- create a python env with python3.11: `python -m venv llm311`
- activate the env: `llm311\Scripts\activate`
- clone this repo: `git clone ... `
- go down to the directory `cd langchain-tuto`
- start vscode: `vscode .`
- install the requirements: `python -m pip install -r requirements.txt`

programs should run !

## first test
simple langchain test on local LLM

# Gen AI RAG Chat bot

Chatbot retrieves specific information from your PDFs using RAG principles, then uses a locally-run LLM via Ollama to generate natural, contextual, and accurate answers to your questions based on that retrieved information.



## Run Locally

Clone the project

```bash
  git clone https://github.com/BJSam/GenAI-ChatBot
```

Go to the project directory

```bash
  cd GenAI-ChatBot
```

Install dependencies

```bash
  uv sync
```

Start the server

```bash
  uv run main.py
```
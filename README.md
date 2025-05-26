# RAG based document question answering model

🦙 **RAG based document question answering model** is a Streamlit-based web application that allows users to upload a PDF document and interactively chat with it using the powerful LLaMA 3.1 model via Groq API.

## 🚀 Features

- Upload and parse PDF documents
- Split documents into manageable chunks
- Create embeddings using `Ollama` embeddings
- Vector storage using `FAISS`
- Query documents using `ConversationalRetrievalChain`
- Maintains conversational memory
- Interactive UI with Streamlit

## 📦 Installation

1. Clone the repository:
```bash

git clone https://github.com/vinaykumarreddy9/LlamaDocChat.git
cd LlamaDocChat

```

2. Create a virtual environment and activate it:
```bash

python -m venv venv
source venv/bin/activate

```

3. Install the required dependencies:
```bash

pip install -r requirements.txt

```

4. Set your Groq API key in a `.env` file:
```

GROQ_API_KEY=your_groq_api_key

```

## 🧠 Usage

```bash

streamlit run app.py

```

Upload a PDF and start chatting with your document powered by LLaMA.

## 📂 Folder Structure

```
RAG_based_document_question_answering_model/
├── app.py
├── .env
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **Groq API**
- **FAISS**
- **Ollama Embeddings**

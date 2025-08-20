
# Customer Support Chatbot

An AI-powered customer support chatbot built with **Meta's LLaMA 2 7B Chat** model, providing real-time, intelligent, and scalable customer service through natural language understanding.

## 🧠 Introduction

This chatbot system is designed to deliver instant, reliable, and context-aware support for customer queries using advanced NLP and retrieval-based techniques. Built using **Streamlit**, **LangChain**, **FAISS**, and **HuggingFace**, the chatbot semantically understands customer messages and fetches accurate responses from a custom support dataset.

---

## 📑 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Dependencies](#-dependencies)
- [Example](#-example)
- [Troubleshooting](#-troubleshooting)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🚀 Features

- 💬 **Conversational Retrieval**: Answers customer questions by retrieving relevant context from a custom CSV dataset.
- ⚡ **Real-time Inference**: Uses `CTransformers` to run the LLaMA 2 model efficiently on local machines.
- 🔍 **Semantic Search**: Leverages FAISS and sentence-transformers for intelligent document search.
- 🌐 **Streamlit UI**: Clean and interactive front-end for chatting with the bot.
- 📁 **Custom Dataset Support**: Trained on `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`.

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/LLM-main.git
cd LLM-main
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install manually:

```bash
pip install streamlit langchain faiss-cpu ctransformers sentence-transformers
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser to [http://localhost:8501](http://localhost:8501)

---

## ⚙️ Configuration

- **Model File**: You must download the `llama-2-7b-chat.ggmlv3.q2_K.bin` file separately and place it in the appropriate path. Update the path in `app.py` if needed.
- **Dataset**: The default CSV file is:
  ```
  Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
  ```

- **FAISS DB Path**:
  ```python
  DB_FAISS_PATH = 'vectorstore/db_faiss'
  ```

---

## 📦 Dependencies

- [LangChain](https://github.com/langchain-ai/langchain)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [CTransformers](https://github.com/marella/ctransformers)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
- [Meta LLaMA 2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

---

## 🧪 Example

Once running, the chatbot UI will prompt you to type a question like:

> "How can I reset my password?"

It will search the support dataset and generate a contextual response using the LLaMA 2 model.

---

## 🛠 Troubleshooting

- ❌ **Model loading error**: Ensure the `.bin` model file is correctly downloaded and the path is valid.
- ❌ **CSV loading failed**: Confirm the dataset exists and is correctly formatted.
- ❌ **Performance issues**: LLaMA 2 7B may require significant RAM/VRAM. Consider using quantized models for efficiency.

---

## 👨‍💻 Contributors

- **Aryan Gupta**  

---

## 📄 License

This project is licensed under the terms of the [LICENSE](./LICENSE) file in this repository.

> © 2025 Aryan Gupta. All rights reserved.

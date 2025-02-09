# Arabic-RAG-Ollama
This project is a **local Retrieval-Augmented Generation (RAG)** system designed to process Arabic PDF documents, perform semantic search, and generate AI-powered answers using the **Ollama 3** model. It is built with Streamlit for the user interface and leverages state-of-the-art NLP models for text embedding and retrieval.
## Main Goal

The goal of this project is to provide a **local, Arabic-language RAG system** that:
1. Processes Arabic PDF files and extracts text.
2. Splits the text into chunks and generates embeddings using a pre-trained Arabic BERT model.
3. Uses FAISS for efficient semantic search over the document chunks.
4. Generates context-aware answers using the **Ollama 3** model, considering both the document content and the conversation history.
5. Provides an interactive interface for users to upload documents, ask questions, and view the conversation history.

## Features

- **Arabic Text Support**: Handles Arabic PDF files and generates answers in Arabic (with optional English translation).
- **Semantic Search**: Uses FAISS for efficient retrieval of relevant document chunks.
- **Conversational Memory**: Maintains a history of the conversation to provide context-aware answers.
- **Expandable History**: Displays the conversation history in an expandable format for easy navigation.
- **Local RAG**: Runs entirely on your local machine, ensuring privacy and control over your data.

## Prerequisites

1. **Python 3.8 or higher**: Ensure Python is installed on your system.
2. **Ollama 3**: Install the Ollama 3 model locally. Follow the instructions below.

### Installing Ollama 3

To use the Ollama 3 model, you need to install it locally. Follow these steps:

1. **Install Ollama**:
   - Visit the [Ollama GitHub repository](https://github.com/ollama/ollama) for installation instructions.
   - Download and install the Ollama CLI tool.

2. **Download the Llama 3 Model**:
   - Run the following command in your terminal to download the Llama 3 model:
     ```bash
     ollama pull llama3
     ```

3. **Verify Installation**:
   - Test the installation by running:
     ```bash
     ollama run llama3
     ```
   - If the model runs successfully, you're ready to use it in this project.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/IlyasNgou/Arabic-RAG-Ollama.git
   cd Arabic-RAG-Ollama

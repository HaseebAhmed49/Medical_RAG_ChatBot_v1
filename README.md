# 📚 Medical RAG ChatBot v1 Documentation

Welcome to the Medical RAG ChatBot v1! This project processes **PDF documents, generates embeddings using HuggingFace models, and stores them in a FAISS vector database** for efficient retrieval-based tasks. It also allows querying the vector database to retrieve relevant information. 🚀

![image](https://github.com/user-attachments/assets/3eaf235d-eedc-4262-a089-210dff769073)

---

## 🌟🚀 Features
✅ **Load PDFs** - The application loads all PDF files from the data directory using the DirectoryLoader.

✅ **Split PDFs into Chunks** - PDFs are split into smaller chunks of 500 characters with a 50-character overlap for better embedding generation.

✅ **Generate Embeddings** - Embeddings are generated using the HuggingFace model: sentence-transformers/all-MiniLM-L6-v2.

✅ **Store in FAISS Vector Database** - The embeddings are stored in a FAISS vector database for efficient retrieval.

✅ **Query the Vector Database** - Users can query the vector database with natural language questions to retrieve relevant information.

---

## Project Structure
```sh
📂 Medical_RAG_ChatBot_v1/
├── 📂 backend/
│   └── main.py          # FastAPI backend for processing and querying embeddings
├── 📂 data/             # Directory for storing PDF files
├── 📂 vectorstore/
│   └── db_faiss/        # FAISS vector database files
├── Pipfile              # Dependency management
├── Pipfile.lock         # Locked dependencies
└── .env                 # Environment variables (optional)
```

---

## Technologies Used
* **Python** Programming Language
* **LangChain** for AI Framework for LLM Applications
* **HuggingFace** AI/ML Hub
* **Mistral** LLM Model
* **FAISS** Vector Database
* **FastAPI** for backend API
* **Streamlit** for ChatBot UI
* **VS Code** (IDE)

---

## ⚙️ Setup Guide
Follow these steps to set up the project on your local machine:

1️⃣ Prerequisites
- 🐍 Python 3.8 or higher
- 📦 Pipenv for dependency management
- 💾 FAISS installed (faiss-cpu)

2️⃣ Clone the Repository
```sh
git clone <repository-url>
cd Medical_RAG_ChatBot_v1
```

3️⃣ Install Dependencies
```sh
pipenv install
```

4️⃣ Activate the Virtual Environment
```sh
pipenv shell
```

5️⃣ Add PDF Files
- Place your PDF files in the data directory.

6️⃣ (Optional) Configure Environment Variables
- Create a .env file in the root directory to store HuggingFace API Token as environment variables.

---

## ▶️ How to Run the Application

### 📌 Backend Start the FastAPI Server
1️⃣ Start the FastAPI Server
Run the following command to start the server:
```sh
uvicorn backend.main:app --reload
```
The server will be available at: http://127.0.0.1:8000

2️⃣ Use the API Endpoints
The application provides two endpoints for interacting with the vector database:

### 🛠️ API Endpoints
1️⃣ POST **/store_embeddings_in_db**
- **Description**: Processes PDF files, generates embeddings for text chunks, and stores them in a FAISS vector database.
- **Steps**:
  1. Load PDF files from the data directory.
  2. Split the PDF content into smaller chunks (500 characters with a 50-character overlap).
  3. Generate embeddings using the HuggingFace model: **sentence-transformers/all-MiniLM-L6-v2**.
  4. Store the embeddings in a FAISS vector database located at db_faiss.
- **Response**:
  * **200 OK**: Embeddings successfully stored.
  * **400 Bad Request**: Missing files or invalid data.
  * **500 Internal Server Error**: Unexpected error.

2️⃣ GET **/query/{question}**
- **Description**: Queries the FAISS vector database with a user-provided question and retrieves relevant information.
- **Steps**:
  1. Load the FAISS vector database and the HuggingFace embedding model.
  2. Use a custom prompt template to structure the query.
  3. Retrieve relevant chunks from the vector database.
  4. Use a language model (e.g., Mistral-7B) to generate an answer based on the retrieved chunks.
- **Path Parameter**:
  * question (string): The question to query the vector database.
- **Response**:
  * **200 OK**: Returns the answer to the query.
  * **404 Not Found**: If no relevant data is found.
  * **500 Internal Server Error**: Unexpected error.

## 🎨 Frontend (Streamlit)

### 📌 Run the Streamlit App
```sh
streamlit run frontend/app.py
```

---

## 🚀 Future Enhancements
- Add more advanced query capabilities (e.g., multi-document summarization).
- Implement user authentication for API endpoints.
- Support additional document formats (e.g., Word, Excel).

---

## 📜 License
This project is licensed under the MIT License.

---

## 📩 Contact
For issues or suggestions, feel free to open an [issue](https://github.com/HaseebAhmed49/Medical_RAG_ChatBot_v1/issues) or reach out!
* Email: haseebahmed02@gmail.com
* LinkedIn/GitHub: /HaseebAhmed49

💡 **Happy Coding!** 🚀

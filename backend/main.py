from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# from dotenv import load_dotenv
import os

app = FastAPI()

#region 1 Endpoint: Store Embeddings
# Step1: Load raw PDFs
DATA_PATH = "data/"
def load_pdfs_from_directory(directory: str):
    """Load PDFs from a directory."""
    loader = DirectoryLoader(
        directory,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    return documents

# Step2: Split PDFs into chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Step3: Create Vector embeddings for each chunk

def get_embedding_model():
    """Load the embedding model."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
    return embedding_model

@app.post("/store_embeddings_in_db")
def store_embeddings_in_vectorstore():
    """Generate embeddings for the text chunks and store them in a vector database."""
    try:
        documents = load_pdfs_from_directory(directory=DATA_PATH)
        if not os.path.exists("data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"):
         raise FileNotFoundError("PDF file not found.")
        print("Length of PDF pages: ", len(documents))

        text_chunks = create_chunks(documents)
        print("Length of PDF chunks: ", len(text_chunks))

        embedding_model = get_embedding_model()

        # Ensure text_chunks is defined or retrieved correctly
        if not text_chunks:
            raise ValueError("text_chunks is empty or not defined.")

        # Step4: Store the embeddings in a vector database
        DB_FAISS_PATH = "vectorstore/db_faiss"
        db = FAISS.from_documents(
            text_chunks,
            embedding_model,
        )
        db.save_local(DB_FAISS_PATH)

        return {"message": "Embeddings stored successfully."}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        # Log the error if logging is setup (optional)
        # logger.error("Failed to store embeddings", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


#endregion

#region 2 Endpoint: Query Embeddings

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.getenv("HF_TOKEN")
print(HF_TOKEN)
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature = 0.1,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": "512",
        }
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )
    return prompt

@app.get("/query/{question}")
def query_vectorstore_db(question: str):
    """Query the vector database with a question and return the answer."""
    try:
        DB_FAISS_PATH = "vectorstore/db_faiss"
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know, dont try to make up an answer.
        Dont provide anything out of the given context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        # Load vector database
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

        # Create QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(HUGGINGFACE_REPO_ID),
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE),
            }
        )

        # Run the query using the URL parameter instead of input()
        response = qa_chain.invoke({'query': question})
        return response

    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail="Vector database not found.")

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
#endregion
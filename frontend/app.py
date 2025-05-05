import os
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"  # Replace with your actual backend URL if deployed

def call_store_embeddings_endpoint():
    try:
        response = requests.post(f"{API_BASE_URL}/store_embeddings_in_db")
        if response.status_code == 200:
            return response.json()["message"]
        else:
            return f"Error: {response.status_code} - {response.json()['detail']}"
    except Exception as e:
        return f"Request failed: {str(e)}"

def query_vectorstore(question):
    try:
        response = requests.get(f"{API_BASE_URL}/query/{question}")
        if response.status_code == 200:
            result = response.json()
            return result.get("result", str(result))  # support old and new formats
        else:
            return f"Error: {response.status_code} - {response.json()['detail']}"
    except Exception as e:
        return f"Request failed: {str(e)}"

def main():
    st.title("🧠 Medical Chatbot with FAISS + Mistral")

    # --- Store Embeddings Section ---
    st.subheader("📚 Load Documents and Store Embeddings")
    if st.button("Store Embeddings in DB"):
        with st.spinner("Storing embeddings..."):
            result = call_store_embeddings_endpoint()
            st.success(result)

    # --- Chat Section ---
    st.subheader("💬 Ask a Question")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Enter your medical query here...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Getting response..."):
            response = query_vectorstore(prompt)

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
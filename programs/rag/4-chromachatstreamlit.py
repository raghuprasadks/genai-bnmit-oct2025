import streamlit as st
import chromadb
import cohere
import os
from dotenv import load_dotenv
#load_dotenv(dotenv_path="../.env")  # Adjust path if needed
load_dotenv()
api_key = os.getenv("cohere_api_key")


def get_relevant_docs(collection, cohere_client, query, top_k=3):
    query_emb = cohere_client.embed(texts=[query]).embeddings[0]
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    docs = results['documents'][0] if results['documents'] else []
    return docs

def generate_answer(cohere_client, query, docs):
    context = "\n".join(docs)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    """
    response = cohere_client.generate(
        prompt=prompt,
        max_tokens=256,
        temperature=0.5
    )
    """

    model = os.getenv("COHERE_MODEL", "command-a-03-2025")
    response = cohere_client.v2.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.5,
    )
    # Extract assistant text from v2 response
    assistant_msg = getattr(response, "message", None)
    if assistant_msg is None or assistant_msg.content is None:
        return ""

    for item in assistant_msg.content:
        if getattr(item, "type", None) == "text":
            return getattr(item, "text", "").strip()

    # Fallback: join any string-like content
    try:
        return "\n".join([c for c in assistant_msg.content if isinstance(c, str)]).strip()
    except Exception:
        return ""

def main():
    st.title("BNMIT Chatbot")
    cohere_api_key = api_key
    cohere_client = cohere.Client(cohere_api_key)
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(name="rag_collection_pdfs")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""

    user_query = st.text_input("You:", value=st.session_state.user_query, key="user_query")
    ask_clicked = st.button("Ask")

    if ask_clicked and user_query:
        docs = get_relevant_docs(collection, cohere_client, user_query)
        answer = generate_answer(cohere_client, user_query, docs)
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("Bot", answer))


    st.header("Chat History")
    for speaker, text in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {text}")

if __name__ == "__main__":
    main()
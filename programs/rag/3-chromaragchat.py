import chromadb
import cohere
import os

from dotenv import load_dotenv

#load_dotenv(dotenv_path="../.env")  # Adjust path if needed
load_dotenv()
api_key = os.getenv("cohere_api_key")

def get_relevant_docs(collection, cohere_client, query, top_k=3):
    # Embed the query
    query_emb = cohere_client.embed(texts=[query]).embeddings[0]
    # Query ChromaDB for similar documents
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    docs = results['documents'][0] if results['documents'] else []
    return docs

def generate_answer(cohere_client, query, docs):
    context = "\n".join(docs)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    # Cohere's legacy `generate` endpoint was removed in 2025; use the v2 Chat API instead.
    # The v2.chat method accepts a list of messages. The message content can be a simple string.
    model = os.getenv("COHERE_MODEL", "command-a-03-2025")
    resp = cohere_client.v2.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.5,
    )

    # The v2 response places assistant text in resp.message.content which is a list of
    # content items; text items have type "text" and a `text` field.
    assistant_msg = resp.message
    if assistant_msg is None or assistant_msg.content is None:
        return ""

    # Find first text content item
    for item in assistant_msg.content:
        if getattr(item, "type", None) == "text":
            return getattr(item, "text", "").strip()

    # Fallback: join any string-like content
    try:
        return "\n".join([c for c in assistant_msg.content if isinstance(c, str)]).strip()
    except Exception:
        return ""

def main():
    cohere_api_key = api_key
    cohere_client = cohere.Client(cohere_api_key)
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(name="rag_collection_pdfs")

    print("RAG Chatbot (type 'exit' to quit)")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        docs = get_relevant_docs(collection, cohere_client, query)
        answer = generate_answer(cohere_client, query, docs)
        print("Bot:", answer)

if __name__ == "__main__":
    main()
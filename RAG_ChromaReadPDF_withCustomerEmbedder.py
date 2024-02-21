import os
import requests
import chromadb
from embedder import CustomEmbedder

class RAGPipeline():
    def __init__(self) -> None:
        self.API_TOKEN = "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" #os.environ["API_TOKEN"]
        self.API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"

    def query(self, prompt):
        headers = {"Authorization": f"Bearer {self.API_TOKEN}"}

        payload = {
            "inputs": prompt,
            "parameters": { #Try and experiment with the parameters
                "max_new_tokens": 50,
                "temperature": 0.1,
                "top_p": 0.9,
                "do_sample": False,
                "return_full_text": False
            }
        }
        response = requests.post(self.API_URL, headers=headers, json=payload)
        return response.json()[0]['generated_text']

    def as_retriever(self, collection, query):
        results = collection.query(
            query_texts=[query],
            n_results=1
        )
        return results['documents']

"""
# Create an instance of the class
rag_pipeline = RAGPipeline()

# Call the query method with a prompt
prompt = "What is the capital of France?"
response = rag_pipeline.query( prompt)

# Print the response
print(response) 

"""
rag_app = RAGPipeline()
custom_embedder = CustomEmbedder()
client = chromadb.PersistentClient(path="../mydb_flblue")
collection = client.get_collection(name="rag_collection_flblue", embedding_function=custom_embedder)

#print(collection.peek( 2 ))

question = "who is ancestor of dogs?"
context = rag_app.as_retriever(collection, question)
prompt = f"""Use the following context to answer the question at the end. Stop when you've answered the question. Do not generate any more than that.
{context}

Question: {question}
"""
print(prompt)
print(rag_app.query(prompt))

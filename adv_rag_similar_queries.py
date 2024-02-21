from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_customllm import CustomLLM

# Load the Llama2 model
model_id = "TheBloke/Llama-2-7B-Chat-GGML"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def generate_similar_queries(original_query):
    # Create a prompt by filling in the original query
    prompt = f"Generate multiple search queries related to: {original_query}"

    llm = CustomLLM()

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    # Generate queries using Llama2
    output = model.generate(input_ids, max_length=100, num_return_sequences=4, no_repeat_ngram_size=2)

    # Decode the generated queries
    generated_queries = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]

    return generated_queries

# Example usage
original_query = "machine learning"
similar_queries = generate_similar_queries(original_query)
print(similar_queries)

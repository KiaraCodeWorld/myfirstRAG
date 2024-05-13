
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_customllm_hf import CustomLLM
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoder

llm = CustomLLM()
query = "What is the difference between a Convolutional Neural Network and a Recurrent Neural Network?"
def hypothetical_answer(query):
    question = query
    template = """
            You are a helpful expert Machine Learning and Deep Learning assistant.
            Provide an example answer to the given question,
            that might be found in a documents like from Book related to machine Learning and Deep Learning.

            The Question about which you have to give example answer is: {question}
            Answer:
    """
    template2 = """
    Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.run(question)
    print(answer)


def generate_multiple_queries(query):
    question = query
    template = """
            You are a helpful expert Machine Learning and Deep Learning assistant.
        Tour Users are asking question related to machine Learning and Deep Learning.
        Suggest up to five additional related questions to help them find the information they need, for the provided question.
        Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic.
        Make sure they are complete questions, and that they are related to the original question.
        Output one question per line. Do not number the questions.

        The question about which you have to generate the question is {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.run(question)
    content = answer.split("\n")
    queries = [query] + content
    filtered_queries = [item.strip() for item in queries if item]
    return filtered_queries


reranker_model = CrossEncoder(model_name="BAAI/bge-reranker-large", max_length=512)

def rerank_docs(query, retrieved_docs):
    query_and_docs = [(query, r.page_content ) for i,r in enumerate(retrieved_docs)]
    scores = reranker_model.predict(query_and_docs)
    return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)


hypothetical_answer(query)
augmented_queries = generate_multiple_queries(query) # + " " + query
#queries = [query] + augmented_queries

print(augmented_queries)


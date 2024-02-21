from langchain_customllm import CustomLLM
from langchain_custom_embedders import CustomChromaEmbedder
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import requests

question = "What are members copays as per the policy?"#"what is difference in in-network and out-of-network services?" #"how to compare the prescription costs?" #"tell me more about florida blue offerd Rewards.?" #"Can you summarize the member policy]=""?" #

member_api2 = """<'member_id': '123456', 'first_name': 'John', 'last_name': 'Doe', 'date_of_birth': '1980-05-15', 'address': <'street': '123 Main St', 'city': 'Anytown', 'state': 'FL', 'zip': '12345'>, 'policies': [<'policy_number': 'P12345', 'policy_type': 'Health Insurance', 'deductible': 1000, 'out_of_pocket_max': 5000, 'copay': <'pcp': 20, 'specialist': 40>, 'coverage_details': <'inpatient_hospital': '80% coverage after deductible', 'outpatient_surgery': '90% coverage after deductible', 'prescription_drugs': 'Tiered copay based on formulary', 'preventive_services': '100% coverage (no copay)'>>, <'policy_number': 'A98765', 'policy_type': 'Dental Insurance', 'deductible': 200, 'out_of_pocket_max': 1000, 'copay': <'pcp': 10, 'specialist': 30>, 'coverage_details': <'routine_cleanings': '100% coverage (no copay)', 'fillings': '80% coverage after deductible', 'major_procedures': '50% coverage after deductible'>>], 'additional_attributes': <'network_type': 'Preferred Provider Organization (PPO)', 'annual_wellness_visit': 'Covered at 100%', 'telehealth_services': 'Available with copay', 'emergency_room': '80% coverage after deductible'>"""
member_api3 = """member policy details are as below : <'member_id': '123456', 'first_name': 'John', 'last_name': 'Doe'  'policies': [<'policy_number': 'P12345', 'policy_type': 'Health Insurance', 'deductible': 1000, 'out_of_pocket_max': 5000, 'copay': <'pcp': 20, 'specialist': 40>, 'coverage_details': <'inpatient_hospital': '80% coverage after deductible', 'outpatient_surgery
'90% coverage after deductible', 'prescription_drugs': 'Tiered copay based on formulary', 'preventive   services': '100% coverage (no copay)'> """

#uvicorn member_api:app --host 0.0.0.0 --port 8001
member_api = requests.get("http://localhost:8001/user").text.replace("{", '<').replace("}", '>')

template = """Instructions: Use only the following context to answer the question. Give very specific answer and dont give explanation when not asked.  
Question: {question}
Context: {context} and """ + f""" member policy details are as below : {member_api}
answer:
"""
#and """ + f""" member policy details are as below : {member_api}

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
llm = CustomLLM()
db = Chroma(persist_directory="./chromadb_flblue3", embedding_function=embedder)

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 1}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
response = qa_chain({'query': question})
print(response['result'])
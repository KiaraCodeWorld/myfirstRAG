import os

from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

"""
An AI is capable of quickly recognizing patterns based off of a few examples. Let's 
instruct our AI to produce a formatted response, where the input format 
is ____, ____ and output format is 'the ____ is ____' . For instance, we may teach the 
LLM to respond in a certain format with examples such as
input: red, car
output: the car is red
This is known as 'few shot' prompting. 'Few shot' prompting is a trick to obtain
more accurate results according to any pattern. In the real world, an AI may use this
pattern to semantically understand some text and extract or classify data using this
pattern.
"""

"""
TODO: Change the prompt below to allow for the generation of output as described above.
You should provide the LLM with a few examples so that the output is correctly 
formatted.
"""
prompt = """Given a sentence, classify it as positive, negative, or neutral.
Text: I love this movie.
Label: Positive
Text: This product is terrible.
Label: Negative
Text: The weather is nice today.
Label: Neutral 
"""

"""
There is no need to change the below function. It will properly use the prompt &
user input as needed.
"""


def llm(user_input):
    llm = HuggingFaceEndpoint(
        endpoint_url=os.environ['LLM_ENDPOINT'],
        task="text2text-generation",
        model_kwargs={
            "max_new_tokens": 200
        }
    )
    chat_model = ChatHuggingFace(llm=llm)
    messages = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(prompt),
        HumanMessagePromptTemplate.from_template("{message}")
    ])

    chain = messages | chat_model
    result = chain.invoke({"message": user_input}).content
    return result

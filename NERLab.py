"""
Named entities are proper nouns that refer to specific entities.
They can be a person, organization, location, date, etc.
For example, in the sentence "Bill Gates and Paul Allen founded Microsoft", we have three named entities.
Bill Gates and Paul Allen are two named entities, of type person, and Microsoft is the third, of type organization.
Named entity recognition, or NER, is the process of extracting Named Entities from the text. It is used in tasks such as classifying content and content recommendation.
"""

"""
We'll be using NLTK's ne_chunk function to detect named entities. 
"""

from nltk import ne_chunk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

def sampleNER():
    ner_text = """
    John Doe, a software engineer at ACME Corporation, recently attended a conference in New York City on January 15-17, 2023. The event, organized by Tech Innovations Inc., focused on artificial intelligence and machine learning. During the conference, John had the opportunity to network with professionals from Google, Microsoft, and other leading tech companies.
    """

    tokens = word_tokenize(ner_text)
    print(tokens)

    pos_tagged = pos_tag(tokens)
    print(pos_tagged)

    result = ne_chunk(pos_tagged)

    print(result)
    # result.draw() #this will open a new window with the tree rendering

"""
Create your own text that includes mentions of named entities of type PERSON and ORGANIZATION and complete the following function to recognize Named Entities for your text. The test will look for the mention of PERSON and ORGANIZATION type named entities. 
"""
def nerExercise():
    ner_text = "Elon musk is owner of Tasla.inc"
    #Tokenize text
    tokens = word_tokenize(ner_text)

    # Tag tokens with POS
    pos_tagged = pos_tag(tokens)
    # use ne_chunk to recognize NER
    result = ne_chunk(pos_tagged)

    return result
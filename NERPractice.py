import nltk
from nltk import ne_chunk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
nltk.download('words')
#ner_text = """
#John Doe, a software engineer at ACME Corporation, recently attended a conference in New York City on January 15-17, 2023. The event, organized by Tech Innovations Inc., focused on artificial intelligence and machine learning. During the conference, John had the opportunity to network with professionals from Google, Microsoft, and other leading tech companies.
#"""


ner_text = "Rami Eid is studying at Stony Brook University in NY"

print("****** word tokenizer ******")
tokens = word_tokenize(ner_text)
print(tokens)

print("****** POS tokenizer ******")
pos_tagged = pos_tag(tokens)
print(pos_tagged)

print("****** word chunks ******")
result = ne_chunk(pos_tagged)

print(result)
result.draw()

text1 = "They refuse to permit us to obtain the refuse big permit" #"Today morning, Arthur felt very good." #"he pluck juicy and big peaches from the tree"
token1 = word_tokenize(text1)
pos_tagged1 = pos_tag(token1)

print(pos_tagged1)



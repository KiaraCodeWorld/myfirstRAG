import nltk
from nltk import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize

#nltk.download('punkt')
nltk.download('wordnet')

text = "The artist decided to create a new painting. Creating art is a form of self-expression. She hoped to create an atmosphere of creativity in her studio where she could freely create. The act of creation brought her joy, and she believed that anyone could create something beautiful with a bit of inspiration."

words = word_tokenize(text)
print("****** tokenized words ******")
print(words)
stemmer = EnglishStemmer()
stemmed_words = [stemmer.stem(word) for word in words]
print("****** stemmed_words ******")
print(stemmed_words)


print("****** lemmatized words ******")
lemmatizer = WordNetLemmatizer()
string_for_lemmatizing = "Can you really have too many pens? They all serve different purposes and one simply cannot have too many!"
words = word_tokenize(string_for_lemmatizing)
#print(words)
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print(lemmatized_words)

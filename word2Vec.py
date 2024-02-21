"""
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
data_corpus=["guru99 is the best sitefor online tutorials. I love to visit guru99."]
vocabulary=vectorizer.fit(data_corpus)
X= vectorizer.transform(data_corpus)
print(X.toarray())
print(vocabulary.get_feature_names_out())

#scikit-learn

# Import the gensim library
import gensim

# Load a pre-trained Word2Vec model
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#gensim.models.KeyedVectors.load_word2vec_format("")
# Find the most similar words to "cat"
print(model.most_similar('cat'))

# Find the word that does not belong in the list
print(model.doesnt_match(['breakfast', 'cereal', 'dinner', 'lunch']))

# Output: 'cereal'

# Find the analogy: king is to queen as man is to ?
print(model.most_similar(positive=['woman', 'king'], negative=['man']))

"""
print("********")
# Import the gensim downloader module
import gensim.downloader as api
from gensim.models import KeyedVectors

# Load the text8 corpus from gensim-data
corpus = api.load('text8')
w2v = api.load('word2vec-google-news-300')

# Print the first 10 words of the corpus
print(list(corpus)[0][:10])
#w2v = KeyedVectors.load_word2vec_format(api.load('word2vec-google-news-300', return_path=True), binary=True)

words = ['cat', 'table']  # provide your own words that you want the results to be similar to
sims = w2v.doesnt_match(words)
print(sims)

#https://huggingface.co/fse/word2vec-google-news-300

#import gensim.downloader as api
#from gensim.models import KeyedVectors
#w2v = KeyedVectors.load_word2vec_format(api.load('word2vec-google-news-300', return_path=True), binary=True)
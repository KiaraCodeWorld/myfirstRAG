"""
Word Embedding
Word embeddings are vectors of words created via an embedding technique. Converting words and sentences to vector embeds allows mathematical operations to be done on them, helping to establish the association of a word with other words with similar meaning. For example, the words such as dog and cat will be placed closer together compared to a word such as 'fire'

Word2Vec
Word2Vec is a word embedding technique that operates under the following assumption: two words sharing similar contexts also share a similar meaning.

It comprises two main architectures: Continuous Bag of Words (CBOW) and Skip-gram.
CBOW predicts a target word based on its context (surrounding words) and Skip-gram predicts the context words given a target word.

The main takeaway here is that Word2Vec is a predictive model focused on the local context.
"""

"""
The library Gensim provides python implementation and already trained models for Word2Vec.
The following model has been trained on the entire Google News dataset, of about 100 billion words.
"""
import gensim.downloader as api

w2v = api.load('word2vec-google-news-300')

def sampleW2V():
    # One of the first things we can do is to find the most frequently appearing words. The following code snippet returns top 10 most frequent words
    print(f"Out of total {len(w2v.index_to_key)} words")
    for index, word in enumerate(w2v.index_to_key):
        if index <= 10:
            print(f"#{index}: {word}")
        else:
            break

    # we can also get a vector representation of a word
    vec_computer = w2v['computer']
    print(vec_computer)

    # However, if it encounters a word it doesn't know, it will throw a KeyError. It is known shortcoming of Word2Vec that it cannot infer vectors of unknown words
    try:
        w2v['revature']  # will throw a KeyError
    except KeyError as e:
        print(e)

# TODO: complete the following function to retrieve vector representation of an input word. You should also make sure if it doesn't know the word, it returns -1, it will not crash.(aka, catch the KeyError)
def retrieveVector(input_word):
    try:
        flag =  w2v[input_word]
    except KeyError as e:
        flag = -1
    return flag


"""
We can also do similarity comparisions using the Word2Vec model.
"""
def sampleSimilarity():
    pairs = [
        # Going from more similar to less similar
        ('cup', 'mug'),
        ('cup', 'bowl'),
        ('cup', 'beverage'),
        ('cup', 'cat'),
    ]
    for w1, w2 in pairs:
        print('%r\t%r\t%.2f' % (w1, w2, w2v.similarity(w1, w2)))

    # It can also retrieve the most similar words to the list of words given. The higher the number, more similar they are in meaning.
    print(w2v.most_similar(positive=['cup', 'mug'], topn=5))

    # The doesnt_match function takes in a list and finds the least similar word.
    print(w2v.doesnt_match(['cup', 'cat', 'mug', 'jar']))


# TODO: Complete the following function so that it calculates the similarity between two words and the similarity index is greater than 0.4
def similarPairExercise():
    word1 = "cat"
    word2 = "feline"

    sim = w2v.similarity(word1, word2)
    if sim > 0.4:
        return sim
    else:
        return -1
    #return  # return w2v similarity index of word 1 and 2 here


# TODO: Complete the following function so that w2v model retrieves top 'max_limit' number of the words you provide
def retrieveSimilarWordsExercise(max_limit: int):
    words = ['cat', 'feline']  # provide your own words that you want the results to be similar to
    sims = w2v.most_similar(positive=words, topn=max_limit)
    return [w for w, s in sims]# grab max_limit number of words similar to words appearing in words


# TODO: Complete the following function to return the word that is the least similar in the given list.
def w2vDoesntMatchExercise(words: list[str]):
    sims = w2v.doesnt_match(words)
    return sims # return the least similar word in the words list
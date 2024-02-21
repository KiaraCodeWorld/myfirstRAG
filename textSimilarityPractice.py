from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        return None

    # Compute the dot product between 2 vectors
    dot_prod = np.dot(vec1, vec2)

    # Compute the norms of the 2 vectors
    norm_vec1 = np.sqrt(np.sum(vec1 ** 2))
    norm_vec2 = np.sqrt(np.sum(vec2 ** 2))

    # Compute the cosine similarity
    cosine_similarity = dot_prod / (norm_vec1 * norm_vec2)

    return cosine_similarity


def cosSimExercise(corpus: list[str]):
    # TODO: Complete this function
    # TODO: Tokenize and vectorize the corpus argument
    vectorizer = CountVectorizer().fit_transform(corpus) #CountVectorizer().fit_transform([corpus]).toarray()
    print(vectorizer)
    # TODO: Calculate cosine similarity
    cosine_sim = cosine_similarity(vectorizer[0, :], vectorizer[1, :])
    # TODO: Return the cosine similarity result of the vectorized text
    return cosine_sim

# Sample texts
text1 = "Natural language processing is fascinating."
text2 = "I'm intrigued by the wonders of natural language processing."

print("*** Vectorizing the texts ***")
# Tokenize and vectorize the texts
vectorizer = CountVectorizer().fit_transform([text1, text2]).toarray()
print(vectorizer)

print(" *** Calculating cosine similarity ***")
# Calculate cosine similarity
cosine_sim = cosine_similarity(vectorizer[0, :], vectorizer[1, :])

# Cosine Similarity ranges from 0 to 1, a number closer to 1 means that they are more similar
print("Cosine Similarity:")
print(cosine_sim)

# This is a python implementation of cosine similarity, using numpy



print("*** calling functions **")

list1 = ['Natural language processing is fascinating', 'I\'m intrigued by the wonders of natural language processing']
print(cosSimExercise(list1))


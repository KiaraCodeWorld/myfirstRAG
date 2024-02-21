"""
Text similarity measures the likeness between two texts, aiming to quantitatively assess the extent of shared content or meaning. Various techniques, including cosine similarity, Euclidean distance, and the Jaccard Index, are employed for different purposes.

Text similarity finds application in diverse areas such as information retrieval, recommendation systems, sentiment analysis, summarization, and plagiarism detection.

In this lab, we will focus on cosine similarity and jaccard similarity index.
"""

"""
Cosine similarity aims to measure the similarity between two texts based on the angle between their respective word vectors. It is often applied to vectors representing the frequency of words in a sentence or document.

To calculate the cosine similarity between two texts, we first represent the texts in vector form. As mentioned earlier, these vectors indicate how frequently certain words are used in a given text. Subsequently, we normalize the vectors to a unit vector, one with a length of 1. Finally, cosine similarity is computed as the dot product of the two vectors, divided by the product of their lengths.

Cosine Similarity is widely used in NLP and information retrieval, particularly in recommendation systems and document classification and clustering.
"""

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


# This is a python implementation of cosine similarity, using numpy
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


def sampleCosineSim():
    # Sample texts
    text1 = "Natural language processing is fascinating."
    text2 = "I'm intrigued by the wonders of natural language processing."

    # Tokenize and vectorize the texts
    vectorizer = CountVectorizer().fit_transform([text1, text2]).toarray()
    print(vectorizer)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(vectorizer[0, :], vectorizer[1, :])

    # Cosine Similarity ranges from 0 to 1, a number closer to 1 means that they are more similar
    print("Cosine Similarity:")
    print(cosine_sim)


# Complete the following function to calculate cosine similarity of two texts.
# It should do the following
# 1. tokenize and vectorize the corpus argument
# 2. Return the cosine similarity result of the vectorized text
# Feel free to play with different text in app.py file, and see if you can find texts that are not at all similar (< .2) or very similar (> .6)!
def cosSimExercise(corpus: list[str]):
    # TODO: Complete this function
    # TODO: Tokenize and vectorize the corpus argument
    vectorizer = CountVectorizer().fit_transform([corpus]).toarray()
    # TODO: Calculate cosine similarity
    cosine_sim = cosine_similarity(vectorizer[0, :], vectorizer[1, :])
    # TODO: Return the cosine similarity result of the vectorized text
    return cosine_sim


"""
Jaccard similarity is a measure of similarity between two sets. It is defined as the size of the intersection divided by the size of the union of the sets. The formula for Jaccard similarity is the size of intersection of two sets over the size of union of two sets.

In NLP, the Jaccard similarity is often used to compare the similarity between two sets of words. For example, it can be used to measure the similarity between two documents based on the sets of words they contain. This is particularly useful in tasks like document clustering, duplicate detection, and information retrieval.
"""


# Jaccard Index is simple to implement using pure python
# We take the number of common words ("intersection") between two sets and divide by the number of all unique words (union) between the two sets.
# The following is a native python implementation of Jaccard Similarity Index
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


from nltk import word_tokenize


# Here is the sample usage of jaccard similarity
def sampleJaccard():
    text1 = "Natural language processing is fascinating."
    text2 = "I'm intrigued by the wonders of natural language processing."

    # Tokenize the sentences and turn them into sets
    set1 = set(word_tokenize(text1))
    set2 = set(word_tokenize(text2))

    # Calculate Jaccard similarity
    jaccard_sim = jaccard_similarity(set1, set2)

    # Print the Jaccard similarity
    print(f"Jaccard Similarity: {jaccard_sim}")


# Use the above jaccard_similarity function to calculate jaccard similarity of two texts in this function.
# Come up with 2 texts that will result in jaccard similarity index of 0.3 or greater
def jaccardExercise():
    # TODO: Complete this function
    text1 = ""
    text2 = ""

    set1 = set(word_tokenize(text1))
    set2 = set(word_tokenize(text2))

    jaccard_sim = jaccard_similarity(set1, set2)

    return jaccard_sim
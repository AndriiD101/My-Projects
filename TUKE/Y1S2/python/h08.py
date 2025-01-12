# -*- coding: utf-8 -*-

# Meno: Denysenko, Andrii
# Spolupráca: 
# Použité zdroje: 
# Čas: 

# Podrobný popis je dostupný na: https://github.com/ianmagyar/introduction-to-python/blob/master/assignments/homeworks/homework08.md

# Hodnotenie: /1b

"""
plot_cosine_similarity - cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space, 
often used in text analysis for comparing documents. If we take vectors with random values and compute their cosine similarity 
with a fixed vector, the result distribution should center around 0 as the number of dimensions increases, 
assuming the elements are from a normal distribution. You should have already completed the compute_cosine_similarity function. 
Use matplotlib to plot the distribution of cosine similarities for a large number of random vectors against a fixed vector. 
Add a vertical line that represents the average cosine similarity. Include axis names and a title in your graph.
"""

import matplotlib.pyplot as plt
import numpy as np


def compute_cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def generate_random_vectors(n, dimensions):
    return np.random.randn(n, dimensions)


def plot_cosine_similarity(fixed_vector, random_vectors):
    cosine_similarities = [compute_cosine_similarity(fixed_vector, v) for v in random_vectors]
    
    plt.hist(cosine_similarities, bins=30, alpha=0.7, color='b')
    
    avg_cosine_similarity = np.mean(cosine_similarities)
    
    plt.axvline(avg_cosine_similarity, color='r', linestyle='dashed', linewidth=2)
    
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cosine Similarities')

    plt.show()


dimensions = 100
n_vectors = 1000
fixed_vector = np.random.randn(dimensions)
random_vectors = generate_random_vectors(n_vectors, dimensions)
cosine_similarities = [compute_cosine_similarity(fixed_vector, v) for v in random_vectors]
print(np.mean(cosine_similarities))
plot_cosine_similarity(fixed_vector, random_vectors)

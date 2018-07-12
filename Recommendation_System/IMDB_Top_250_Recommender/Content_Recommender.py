# PhD Student: Nguyen Dang Quang, Computer Science Department, KyungHee University, Korea.
# Build based on tutorial in Datacamp https://www.datacamp.com/community/tutorials/recommender-systems-python
# Credit to the author Rounak Banik of the tutorial.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

metadata1 = pd.read_csv('movies_metadata.csv', low_memory=False)
metadata = metadata1.copy().loc[1:3000]

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

# Construct the required IF-IDF matrix by filtering and transforming the data
tfidf_maxtrix = tfidf.fit_transform(metadata['overview'])
# Output the shape of tdidf_maxtrix
print(tfidf_maxtrix.shape)

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_maxtrix, tfidf_maxtrix)
# Output the shape of cosine_sim
print(cosine_sim.shape)

# Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title'])


# Find the similar movie
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches to the title
    idx = indices.loc[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[0:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    return metadata['title'].iloc[movie_indices]


print(get_recommendations('Jumanji'))

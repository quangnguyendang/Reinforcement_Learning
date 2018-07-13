# PhD Student: Nguyen Dang Quang, Computer Science Department, KyungHee University, Korea.
# Built with some bug fixed based on tutorial in Datacamp
# https://www.datacamp.com/community/tutorials/recommender-systems-python
# Credit to the author Rounak Banik of the tutorial.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import ast

metadata1 = pd.read_csv('movies_metadata.csv', low_memory=False)
metadata = metadata1.copy().loc[1:30]

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

    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    return metadata['title'].iloc[movie_indices]


# Load keywords and credits
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

# Convert IDs to int. Required for merging
keywords['id'] = keywords['id'].astype('int')
credits[id] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

print('Recommendation for things similar to Jumanji movie')
print(get_recommendations('Jumanji', cosine_sim))


def get_director(x):
    for i in ast.literal_eval(x):
        if i['job'] == 'Director':
            print(i['job'] + ": " + i['name'])
            return i['name']
    return np.nan


# Return the list of top 3 elements or entire list; whichever is more
def get_list(x1):
    x = ast.literal_eval(x1)
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

# Print metadata for the first 5 films
with pd.option_context('display.max_rows', None, 'display.max_columns', 6):
    print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(5))



# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

# Print metadata for the first 5 films
with pd.option_context('display.max_rows', None, 'display.max_columns', 6):
    print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(5))


# Create metadata soup
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


metadata['soup'] = metadata.apply(create_soup, axis=1)
# with pd.option_context('display.max_rows', None, 'display.max_columns', 6):
#     print(metadata[['title', 'soup']].head(5))

# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])

# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of your main DataFrame and construct reverse mapping as before
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])

print(get_recommendations('Jumanji', cosine_sim2))

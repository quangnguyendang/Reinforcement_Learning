# PhD Student: Nguyen Dang Quang, Computer Science Department, KyungHee University, Korea.
# Build based on tutorial in Datacamp https://www.datacamp.com/community/tutorials/recommender-systems-python
# Credit to the author Rounak Banik of the tutorial.

import pandas as pd

metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Calculate average vote of entire dataset - C
C = metadata['vote_average'].mean()

# Calculate the minimum number of votes requires to be in the charts - m
m = metadata['vote_count'].quantile(0.9)


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(v+m) * C)


q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)

print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))



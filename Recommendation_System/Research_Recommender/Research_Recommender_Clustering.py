# Example of CBF for research-paper domain
# Nguyen Dang Quang

from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from nltk.corpus import stopwords

# --------------------------------------------------------
user_input_data = "It is known that the performance of an optimal control strategy obtained from an off-line " \
                  "computation is degraded under the presence of model mismatch. In order to improve the control " \
                  "performance, a hybrid neural network and on-line optimal control strategy are proposed in this " \
                  "study and demonstrated for the control of a fed-batch bioreactor for ethanol fermentation. The " \
                  "information of the optimal feed profile of the fed-batch reactor. The simulation results show " \
                  "that the neural network provides a good estimate of unmeasured variables and the on-line optimal " \
                  "control with the neural network estimator gives a better control performance in terms of the " \
                  "amount of the desired ethanol product, compared with a conventional off-line optimal control " \
                  "method."

user_title = "user undefined title"

# --------------------------------------------------------

metadata = pd.read_json('sample-records', lines=True)

user_data = pd.DataFrame([[user_input_data, user_title]], columns=['paperAbstract', 'title'])
metadata = pd.concat([metadata, user_data], sort=True).fillna('')

filter_na = metadata["paperAbstract"] != ''
metadata = metadata[filter_na]


# Lower the characters
def clean_data(x):
    if isinstance(x, list):
        list_data = []
        for i in x:
            list_data.append(str.lower(str(i)))
        return list_data
    elif isinstance(x, str):
        return str.lower(str(x))
    else:
        return ' '


# turn list of string items into string
def get_string(x):
    if isinstance(x, list):
        names = ''
        for i in x:
            names = names + i['name'] + " "
        return names
    else:
        return str(x)


# turn list of entity string items into string
def get_entity(x):
    if isinstance(x, list):
        names = ''
        for i in x:
            names = names + str(i) + " "
        return names
    else:
        return str(x)


# Apply clean_data function to your features.
features = ['authors', 'title', 'journalName', 'paperAbstract']

for feature in features:
    metadata[feature] = metadata[feature].apply(get_string)
    metadata[feature] = metadata[feature].apply(clean_data)

metadata['entities'] = metadata['entities'].apply(get_entity)


# Create metadata soup
def create_soup(x):
    return x['journalName'] + ' ' + x['title'] + ' ' + x['title'] + ' ' + x['paperAbstract'] + ' ' + x['entities'] + ' ' + x['entities'] + ' ' + x['entities']


metadata['soup'] = metadata.apply(create_soup, axis=1)

# --------------------------------------------------------
stemmer = SnowballStemmer("english")


def word_stem_and_stopword_remove(x1):
    x = x1['soup']
    final = ''
    for y in x.split(' '):
        if y not in stopwords.words('english'):
            final = final + stemmer.stem(y) + ' '
    return final


metadata['filtered'] = metadata.apply(word_stem_and_stopword_remove, axis=1)

# Print metadata for the first 5 films
with pd.option_context('display.max_rows', None, 'display.max_columns', 10, 'display.max_colwidth', 100):
    print(metadata[['soup', 'filtered']].head(5))

print('\n\n Done Pre-processing Data \n\n')

# --------------------------------------------------------
# TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tvec = TfidfVectorizer(min_df=1, max_df=10, stop_words='english', ngram_range=(1, 2))
tvec_weights = tvec.fit_transform(metadata.filtered.dropna())

# --------------------------------------------------------
# Classifier for User's Text - K-MEAN - http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=24, random_state=0).fit(tvec_weights)

# print('metadata shape = ', metadata.filtered.shape)
# print('k-mean shape = ', kmeans.labels_.shape)

metadata['cluster_number_kmean'] = kmeans.labels_

# User Data and similar papers
def find_cluster_data_kmean(title):
    print('\n\nSimilar papers using K-mean Clustering: \n')
    filter_title = metadata['title'] == title
    user_cluster = str(metadata.loc[filter_title].cluster_number_kmean.item())
    similar_papers = metadata.loc[metadata['cluster_number_kmean'] == int(user_cluster)].title
    with pd.option_context('display.max_rows', None, 'display.max_columns', 10, 'display.max_colwidth', -1):
        print(similar_papers)


find_cluster_data_kmean(user_title)

# --------------------------------------------------------
# Classifier for User's Text - Birch - http://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch

from sklearn.cluster import Birch

brc = Birch(branching_factor=50, n_clusters=24, threshold=0.5, compute_labels=True)
brc.fit(tvec_weights)

metadata['cluster_number_birch'] = brc.labels_

# User Data and similar papers
def find_cluster_data_birch(title):
    print('\n\nSimilar papers using Birch Clustering: \n')
    filter_title = metadata['title'] == title
    user_cluster = str(metadata.loc[filter_title].cluster_number_birch.item())
    similar_papers = metadata.loc[metadata['cluster_number_birch'] == int(user_cluster)].title
    with pd.option_context('display.max_rows', None, 'display.max_columns', 10, 'display.max_colwidth', -1):
        print(similar_papers)


find_cluster_data_birch(user_title)

# --------------------------------------------------------
# Classifier for User's Text - Agglomerative Clustering - http://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

from sklearn.cluster import AgglomerativeClustering


# User Data and similar papers
def find_cluster_data_Agglomerative(title):
    print('Similar papers using Agglomerative Clustering: \n')
    filter_title = metadata['title'] == title
    user_cluster = str(metadata.loc[filter_title].cluster_number_Agglomerative.item())
    similar_papers = metadata.loc[metadata['cluster_number_Agglomerative'] == int(user_cluster)].title
    with pd.option_context('display.max_rows', None, 'display.max_columns', 10, 'display.max_colwidth', -1):
        print(similar_papers)


for linkage in ('ward', 'average', 'complete'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=24)
    clustering.fit(tvec_weights.toarray())
    metadata['cluster_number_Agglomerative'] = clustering.labels_
    print('\n\nAgglomerative Clustering - ', linkage)
    find_cluster_data_Agglomerative(user_title)

# --------------------------------------------------
# COSINE SIMILARITY
from sklearn.metrics.pairwise import cosine_similarity


# Find the similar movie
def get_recommendations(title, cosine_sim):
    # Get the index of the paper that matches to the title
    idx = indices.loc[title]
    # Get the pairwise similarity scores of all paper with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the papers based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:5]
    # Get the paper indices
    paper_indices = [i[0] for i in sim_scores]
    return metadata['title'].iloc[paper_indices]


cosine_sim = cosine_similarity(tvec_weights, tvec_weights)
indices = pd.Series(metadata.index, index=metadata['title'])

print('\n\nSimilar paper using Cosine Similarity: \n')
with pd.option_context('display.max_rows', None, 'display.max_columns', 10, 'display.max_colwidth', -1):
    print(get_recommendations(user_title, cosine_sim))

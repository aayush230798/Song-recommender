import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def recommend(song):
    index = songs[songs['Song-Artist'] == song].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_songs = []
    for i in distances[1:8]:
        recommended_songs.append(songs.iloc[i[0]]['Song-Artist'])
    return recommended_songs


songs_dict = pickle.load(open('songs_dict.pkl', 'rb'))
songs = pd.DataFrame(songs_dict)

complete_feature_df = pickle.load(open('feature_set_final.pkl', 'rb'))
X = complete_feature_df.drop(['id'], axis=1).to_numpy()

similarity = cosine_similarity(X)

st.title('Song Recommender System')

selected_song_name = st.selectbox(
    'Which song you want more songs to be recommended for?',
    np.append(songs['Song-Artist'].values,[""], axis = 0),index = len(songs))


if st.button('RECOMMEND'):
    recommendations = recommend(selected_song_name)
    for i in recommendations:
        st.write(i)

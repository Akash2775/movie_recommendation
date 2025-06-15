import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load your movie dataset
@st.cache_data
def load_data():
    return pd.read_csv("movies_metadata.csv")  # Make sure this CSV exists in your project directory

movies = load_data()

# Check actual columns to help select features
st.sidebar.write("Dataset Columns:", movies.columns)

# Sidebar settings
st.sidebar.title("Movie Recommendation Settings")

# 🟢 Automatically select numerical columns (except movie title or other non-numeric ones)
numeric_columns = movies.select_dtypes(include=np.number).columns.tolist()

if not numeric_columns:
    st.error("No numeric features found in your dataset. Please preprocess the data to include numeric features.")
    st.stop()

# You can also manually select features from numeric columns
feature_columns = st.sidebar.multiselect("Select features for clustering:", numeric_columns, default=numeric_columns)

features = movies[feature_columns].dropna()

# Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
movies = movies.loc[features.index]  # Align original dataset with features
movies['Cluster'] = kmeans.fit_predict(scaled_features)

# Main App
st.title("🎬 Movie Recommendation System")

if 'title' in movies.columns:
    movie_title_column = 'title'
elif 'movie_title' in movies.columns:
    movie_title_column = 'movie_title'
else:
    movie_title_column = st.sidebar.selectbox("Select movie title column:", movies.columns)

selected_movie = st.selectbox("Select a Movie:", movies[movie_title_column].dropna().unique())

if st.button("Recommend Similar Movies"):
    try:
        selected_cluster = movies[movies[movie_title_column] == selected_movie]['Cluster'].values[0]
        recommended_movies = movies[(movies['Cluster'] == selected_cluster) & (movies[movie_title_column] != selected_movie)]
        st.subheader("Recommended Movies:")
        for title in recommended_movies[movie_title_column].head(10):
            st.write(f"🎥 {title}")
    except IndexError:
        st.error("Selected movie not found in clustered dataset.")

# Optional: Show dataset preview
if st.checkbox("Show Dataset"):
    st.dataframe(movies.head(20))

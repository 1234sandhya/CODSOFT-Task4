from sklearn.metrics.pairwise import cosine_similarity
ratings_data = {
    'userId': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
    'movieId': [1, 2, 3, 1, 4, 2, 3, 5, 1, 2, 3, 5],
    'rating': [5, 4, 3, 4, 5, 5, 4, 3, 2, 4, 5, 5]
}

ratings = pd.DataFrame(ratings_data)
ratings.to_csv('ratings.csv', index=False)

movies_data = {
    'movieId': [1, 2, 3, 4, 5],
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E']
}

movies = pd.DataFrame(movies_data)
movies.to_csv('movies.csv', index=False)

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

data = pd.merge(ratings, movies, on='movieId')

user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

user_movie_matrix.fillna(0, inplace=True)

user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def get_user_similar_movies(user_id, num_recommendations=5):
    # Get the user's ratings
    user_ratings = user_movie_matrix.loc[user_id]

    similar_users = user_similarity_df[user_id]

    similar_users = similar_users.sort_values(ascending=False)

  
    weighted_ratings = pd.Series(0, index=user_movie_matrix.columns)
    for i in range(1, len(similar_users)):
        similar_user_id = similar_users.index[i]
        similar_user_ratings = user_movie_matrix.loc[similar_user_id]
        weighted_ratings += similar_user_ratings * similar_users.iloc[i]

    watched_movies = user_ratings[user_ratings > 0].index
    weighted_ratings = weighted_ratings.drop(watched_movies, errors='ignore')

    recommendations = weighted_ratings.sort_values(ascending=False).head(num_recommendations)
    return recommendations.index.tolist()

recommended_movies = get_user_similar_movies(1)
print(f"Recommended movies for user 1: {recommended_movies}")

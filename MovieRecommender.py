"""
Movie Recommendation System using Neural Network Regression

This script implements a movie recommendation system that:
1. Processes movie metadata (actors, directors, genres, tags)
2. Creates movie profiles using TF-IDF vectorization
3. Builds user-movie rating matrix from training data
4. Uses MLP Regressor to predict ratings for test data
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize
import numpy as np


def load_tags():
    """Load and process tags data."""
    tags = {}
    print("Processing tags...")
    
    with open("tags.dat", "r") as tags_file:
        first_line_done = False
        for line in tags_file:
            if first_line_done:
                parts = line.split('\t')
                tags[parts[0]] = parts[1]
            else:
                first_line_done = True
    
    print("Done")
    return tags


def process_actors():
    """Process movie actors data and create initial movie profiles."""
    print("Processing actors...")
    movie_id_index = []
    movie_profiles = []
    
    with open("movie_actors.dat", "r", errors='ignore') as actors_file:
        first_line_done = False
        current_movie_id = 0
        
        for line in actors_file:
            parts = line.split('\t')
            if first_line_done:
                if parts[0] != current_movie_id:  # New movie found
                    movie_id_index.append(parts[0])
                    current_movie_id = parts[0]
                    movie_profiles.append("")
                    if int(parts[3]) < 5:  # Only add top 5 actors
                        movie_profiles[-1] += parts[1] + " "
                else:
                    if int(parts[3]) < 5:
                        movie_profiles[-1] += parts[1] + " "
            else:
                first_line_done = True
    
    return movie_id_index, movie_profiles


def add_directors(movie_id_index, movie_profiles):
    """Add directors information to movie profiles."""
    print("Processing directors...")
    
    with open("movie_directors.dat", "r", errors='ignore') as directors_file:
        first_line_done = False
        
        for cnt, line in enumerate(directors_file):
            parts = line.split('\t')
            if first_line_done:
                if cnt - 1 < len(movie_id_index) and movie_id_index[cnt - 1] == parts[0]:
                    movie_profiles[cnt - 1] += parts[1] + " "
                else:
                    for i, movie_id in enumerate(movie_id_index):
                        if movie_id == parts[0]:
                            movie_profiles[i] += parts[1] + " "
                            break
            else:
                first_line_done = True


def add_genres(movie_id_index, movie_profiles):
    """Add genres information to movie profiles."""
    print("Processing genres...")
    
    with open("movie_genres.dat", "r") as genres_file:
        first_line_done = False
        
        for line in genres_file:
            parts = line.split('\t')
            if first_line_done:
                for i, movie_id in enumerate(movie_id_index):
                    if movie_id == parts[0]:
                        movie_profiles[i] += parts[1].rstrip() + " "
                        break
            else:
                first_line_done = True


def add_tags(movie_id_index, movie_profiles, tags):
    """Add tags information to movie profiles."""
    print("Processing tags...")
    
    with open("movie_tags.dat", "r") as movie_tags_file:
        first_line_done = False
        tag_id_list = []
        tag_weight_list = []
        current_movie_id = "1"
        
        for line in movie_tags_file:
            parts = line.split('\t')
            if first_line_done:
                if parts[0] != current_movie_id:
                    # Process accumulated tags for previous movie
                    for tag_idx, tag_id in enumerate(tag_id_list):
                        for movie_idx, movie_id in enumerate(movie_id_index):
                            if movie_id == parts[0]:
                                weight = int(tag_weight_list[tag_idx])
                                tag_text = tags[tag_id].rstrip().replace(" ", "_")
                                movie_profiles[movie_idx] += (tag_text + " ") * weight
                                break
                    tag_id_list = []
                    tag_weight_list = []
                    current_movie_id = parts[0]
                
                tag_id_list.append(parts[1])
                tag_weight_list.append(parts[2].rstrip())
            else:
                first_line_done = True


def create_user_profiles(movie_id_index, movie_profiles):
    """Create user profiles from training data."""
    print("Creating utility matrix...")
    user_id_index = []
    user_profiles = []
    
    with open("train.dat", "r") as train_file:
        first_line_done = False
        first_user_done = False
        current_user_id = "0"
        current_movie_count = 0
        
        for cnt, line in enumerate(train_file):
            if cnt % 10000 == 0:
                print(f"Processed {cnt} lines")
            
            parts = line.split(' ')
            if first_line_done:
                if parts[0] != current_user_id:  # New user found
                    if first_user_done:
                        # Fill remaining movies with 0s for previous user
                        while current_movie_count < len(movie_profiles):
                            user_profiles[-1] += "0 "
                            current_movie_count += 1
                    
                    user_id_index.append(parts[0])
                    current_user_id = parts[0]
                    user_profiles.append("")
                    current_movie_count = 0
                    
                    # Add ratings for current user
                    for i, movie_id in enumerate(movie_id_index):
                        if movie_id == parts[1]:
                            user_profiles[-1] += parts[2].rstrip() + " "
                            current_movie_count = i + 1
                            break
                        else:
                            user_profiles[-1] += "0 "
                    first_user_done = True
                else:
                    # Continue with same user
                    for i, movie_id in enumerate(movie_id_index[current_movie_count:], current_movie_count):
                        if parts[1] not in movie_id_index:
                            break
                        if movie_id == parts[1]:
                            user_profiles[-1] += parts[2].rstrip() + " "
                            current_movie_count = i + 1
                            break
                        else:
                            user_profiles[-1] += "0 "
            else:
                first_line_done = True
        
        # Fill remaining movies with 0s for last user
        while current_movie_count < len(movie_profiles):
            user_profiles[-1] += "0 "
            current_movie_count += 1
    
    print("Done")
    return user_id_index, user_profiles


def convert_user_profiles_to_arrays(user_profiles):
    """Convert user profiles from strings to numpy arrays."""
    print("Converting user profiles to arrays...")
    user_profiles_arrays = []
    
    for user_profile in user_profiles:
        ratings = user_profile.split(' ')[:-1]  # Remove last empty element
        rating_array = np.array([float(rating) for rating in ratings])
        user_profiles_arrays.append(rating_array)
    
    return user_profiles_arrays


def predict_ratings(movie_id_index, movie_profiles, user_id_index, user_profiles_arrays):
    """Generate predictions for test data using MLP Regressor."""
    print("Generating predictions...")
    
    clf = MLPRegressor(max_iter=50, solver='lbfgs', hidden_layer_sizes=15)
    vectorizer = TfidfVectorizer()
    
    with open("test.dat", "r") as test_file, open("result.dat", "w") as result_file:
        first_line_done = False
        
        for cnt, line in enumerate(test_file):
            if cnt % 100 == 0:
                print(f"Processed {cnt} test cases")
            
            parts = line.split(' ')
            if first_line_done:
                movie_id = parts[1].rstrip()
                user_id = parts[0]
                
                if movie_id not in movie_id_index:
                    print(f"Warning: Unknown movie ID {movie_id}")
                    result_file.write("3.0\n")
                elif user_id not in user_id_index:
                    print(f"Warning: Unknown user ID {user_id}")
                    result_file.write("3.0\n")
                else:
                    movie_index = movie_id_index.index(movie_id)
                    user_index = user_id_index.index(user_id)
                    
                    movie_profile = movie_profiles[movie_index]
                    user_ratings = user_profiles_arrays[user_index]
                    
                    # Get training data for this user (non-zero ratings)
                    training_movies = []
                    training_ratings = []
                    for i, rating in enumerate(user_ratings):
                        if rating != 0.0:
                            training_movies.append(movie_profiles[i])
                            training_ratings.append(rating)
                    
                    if training_movies:
                        # Vectorize and normalize
                        X_train = vectorizer.fit_transform(training_movies)
                        X_train = normalize(X_train)
                        X_test = vectorizer.transform([movie_profile])
                        
                        # Train and predict
                        y_train = np.array(training_ratings, dtype=np.float64)
                        clf.fit(X_train, y_train)
                        predicted_rating = clf.predict(X_test)[0]
                        
                        # Clamp rating to valid range
                        predicted_rating = round(min(max(predicted_rating, 1.0), 5.0), 1)
                        result_file.write(f"{predicted_rating}\n")
                    else:
                        result_file.write("3.0\n")
            else:
                first_line_done = True


def main():
    """Main function to orchestrate the movie recommendation process."""
    print("Movie Recommender")
    print("=" * 50)
    
    # Load and process data
    tags = load_tags()
    movie_id_index, movie_profiles = process_actors()
    add_directors(movie_id_index, movie_profiles)
    add_genres(movie_id_index, movie_profiles)
    add_tags(movie_id_index, movie_profiles, tags)
    
    print(f"\nExample movie profile:\n{movie_profiles[-2]}")
    print("Movie profile creation complete\n")
    
    # Create user profiles
    user_id_index, user_profiles = create_user_profiles(movie_id_index, movie_profiles)
    user_profiles_arrays = convert_user_profiles_to_arrays(user_profiles)
    
    print(f"Total movies: {len(movie_profiles)}")
    print(f"Total users: {len(user_profiles)}")
    
    # Generate predictions
    predict_ratings(movie_id_index, movie_profiles, user_id_index, user_profiles_arrays)
    
    print("Prediction complete!")


if __name__ == "__main__":
    main()


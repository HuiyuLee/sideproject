import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.engine import data_adapter
from sklearn.metrics.pairwise import cosine_similarity


def read_csv_file(file_name):
    # Get current .py file and 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get data full path
    file_path = os.path.join(current_dir, file_name)
    print(f"Full path: {file_path}")
    try:
        # Check the file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found in {file_path}. Please check the file path and name.")
        # read the data and store data in DataFrame
        data = pd.read_csv(file_path)
        # print a summary of the data
        print("File read successfully! Data Info:")
        print(data.info())
        #print(data.describe())

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("File not found.")

    except pd.errors.EmptyDataError:
        print("Error: File is empty. Please check the file path and name.")

    except pd.errors.ParserError:
        print("Error: Failed to parse the CSV file. Please check the file format.")

    except Exception as e:
        print(f"Error: {e}")

    return data

def handle_missing_data(data):
    """Check and remove rows with missing values from the dataset."""
    # Remove rows with missing values
    cleaned_data = data.dropna()

    return cleaned_data

def one_hot_encode(data, column, sep):
    data_dummies = data[column].str.get_dummies(sep)
    features = pd.concat([data, data_dummies], axis=1)

    return features, data_dummies.columns

def convert_obj2num_columns(data, columns):
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    return data

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

def convert_runtime(runtime):
    # convert 2h22min
    match = re.match(r'(\d+)h (\d+)m', str(runtime))
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    # convert 2h
    match = re.match(r'(\d+)h', str(runtime))
    if match:
        return int(match.group(1)) * 60
    # convert 30min
    match = re.match(r'(\d+)m', str(runtime))
    if match:
        return int(match.group(1))
    # Return NaN if convert unsuccessful
    return np.nan

def get_similar_movies(movie_idx, embeddings, top_k=5):
    """Return similar movies index
    movie_idx: input movies index in embeddings
    embeddings: all moviesembeddings
    """
    # Calculates the cosine similarity between a specific movie and all movies in the dataset.
    similarities = cosine_similarity([embeddings[movie_idx]], embeddings)[0]
    # Sorts the similarity scores in descending order and removes the movie itself (cosine similarity of 1) 
    similar_indices = similarities.argsort()[::-1][1:top_k+1]
    return similar_indices


# Main script to execute the function
if __name__ == "__main__":

    """ parameter """
    # the data file name
    file_name = 'IMDB Top 250 Movies.csv'
    encoding_dim = [8, 16]

    """Data """
    # Read CSV File
    movies_data = read_csv_file(file_name)
    # Remove rows with missing values from the dataset
    cleaned_movies_data = handle_missing_data(movies_data)
    # One-Hot Encoded
    movies_features, genre_dummies = one_hot_encode(cleaned_movies_data, column='genre', sep=',')
   # Budget: Not Available to NaN
    movies_features['budget'] = movies_features['budget'].replace('Not Available', np.nan)
    # Object to number
    movies_features = convert_obj2num_columns(movies_features, ['budget'])
    # Transfer nan to float32
    movies_features['budget'] = movies_features['budget'].fillna(0)  # 用 0 或其他適當數值填補

    # Rum time convert to time number
    movies_features['run_time'] = movies_features['run_time'].apply(convert_runtime)
    movies_features['run_time'] = movies_features['run_time'].fillna(movies_features['run_time'].median())

    # Add 'budget', 'run_time', 'rating', 'year' as feature
    feature_cols = ['budget', 'run_time', 'rating', 'year'] + list(genre_dummies)
    movie_input_features = movies_features[feature_cols]
    features_array = movie_input_features.values


    """Autoencoder Model"""
    input_dim = movie_input_features.shape[1] # 25
    # Set input layer
    input_layer = Input(shape=(input_dim,)) # 1-D tuple, if shape=(input_dim) means a number 
    # Encoder：Feature Compression
    encoded = Dense(encoding_dim[1], activation='relu')(input_layer)
    # Decoder：Reconstructing the original features
    decoded = Dense(input_dim, activation='linear')(encoded)
    # Set Autoencoder model
    data_adapter._is_distributed_dataset = _is_distributed_dataset
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()

    autoencoder.fit(features_array, features_array, epochs=50, batch_size=16, shuffle=True, validation_split=0.2)

    encoder = Model(inputs=input_layer, outputs=encoded)
    movie_embeddings = encoder.predict(features_array)
    print("Movie embeddings shape:", movie_embeddings.shape)

    # Test: choose the first movie
    similar_movies = get_similar_movies(0, movie_embeddings, top_k=5)
    print("Indices of similar movies:", similar_movies)
    # Get the name of movies from similar index
    similar_movie_names = [movies_data.iloc[i]['name'] for i in similar_movies]
    print("Names of similar movies:", similar_movie_names)

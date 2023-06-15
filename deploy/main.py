from flask import Flask, jsonify, request
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from fatsecret import Fatsecret
import random
from tensorflow import keras
import numpy as np
import pandas as pd 
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

app = Flask(__name__)
cred = credentials.Certificate("api.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Set up FatSecret API credentials
consumer_key = '47cd76bdcb1a4a47b1d8fe9b5c619519'
consumer_secret = 'a85a7efd87d7448089fa7fb93a7f175b'
fs = Fatsecret(consumer_key, consumer_secret)


@app.route('/',  methods=['GET'])
def test():
    return 'connected'

# /food-detail?food_id=<food_id>
@app.route('/food-details', methods=['GET'])  # Add this line
def food_details():
    food_id = request.args.get('food_id')
    food = fs.food_get(food_id)

    # check the response
    #print(food)  
    
    if food and 'servings' in food:
        servings = food['servings']['serving']
        
        if isinstance(servings, list):
            # Get the first serving 
            serving = servings[0]  
        else:
            # when only one serving
            serving = servings  
        
        food_details = {
            'food_id': food['food_id'],
            'food_name': food['food_name'],
            'food_url': food['food_url'],
            'calories': serving['calories'],
            'fat': serving['fat'],
            'carbs': serving['carbohydrate'],
            'protein': serving['protein'],
            'sodium': serving['sodium'],
            'iron': serving['iron'],
            'cholesterol': serving['cholesterol']
            # Add on
        }
        
        return jsonify(food_details)
    
    return jsonify({'message': 'Food not found'})

# food search on searchbar menu /food-search?query=<food_name>
@app.route('/food-search', methods=['GET'])
def search_food():
    query = request.args.get('query')
    foods = fs.foods_search(query)
    
    if foods:
        food_info = []
        for food in foods:
            food_item = {
                'food_id': food['food_id'],
                'food_name': food['food_name'],
                # 'food_description': food['food_description'],
                'calories': food['food_description'].split(' | ')[0].split(': ')[1],
                'fat': food['food_description'].split(' | ')[1].split(': ')[1],
                'carbs': food['food_description'].split(' | ')[2].split(': ')[1],
                'protein': food['food_description'].split(' | ')[3].split(': ')[1]
            }
            food_info.append(food_item)
        
        return jsonify(food_info)
    
    return jsonify({'message': 'Food not found'})

# food search random for first page of search food
@app.route('/food-search-menu', methods=['GET'])
def search_food_menu():
    # randomize search
    foods = ['sapi', 'daging', 'ayam', 'susu', 'nasi']
    query = random.choice(foods)
    foods = fs.foods_search(query)
    
    if foods:
        food_info = []
        for food in foods:
            food_item = {
                'food_id': food['food_id'],
                'food_name': food['food_name'],
                'calories': food['food_description'].split(' | ')[0].split(': ')[1],
                'fat': food['food_description'].split(' | ')[1].split(': ')[1],
                'carbs': food['food_description'].split(' | ')[2].split(': ')[1],
                'protein': food['food_description'].split(' | ')[3].split(': ')[1]
            }
            food_info.append(food_item)
        
        return jsonify(food_info)
    
    return jsonify({'message': 'Food not found'})

# get document user
@app.route('/documents/<id>', methods=['GET'])
def get_document(id):
    try:
        doc_ref = db.collection('users').document(id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({'message': 'document not found'}), 404
        data = doc.to_dict()
        return jsonify(data), 200
    except Exception as e:
        print('error retrieving document:', e)
        return jsonify({'message': 'internal server error'}), 500
    
# add favorite food
@app.route('/love', methods=['POST'])
def add_favorite_food():
    try:
        data = request.get_json()
        user_id = data['userId']
        item_data = data['itemData']
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            user_ref.set({'favorite': [item_data]})
        else:
            user_ref.update({'favorite': firestore.ArrayUnion([item_data])})

        return 'item added to favorites', 200
    except Exception as e:
        print('Error transferring item:', e)
        return 'error transferring item', 500

# delete favorite food
@app.route('/users/<userId>/favorite/<foodId>', methods=['DELETE'])
def delete_favorite_food(userId, foodId):
    try:
        user_ref = db.collection('users').document(userId)
        user_doc = user_ref.get()

        if not user_doc.exists:
            return jsonify({'error': 'user not found'}), 404
        favorite_food = user_doc.to_dict().get('favorite', [])
        food_index = next((index for index, food in enumerate(favorite_food) if food.get('food_id') == foodId), -1)
        if food_index == -1:
            return jsonify({'error': 'food not found in favorite food list'}), 404
        del favorite_food[food_index]
        user_ref.update({'favorite': favorite_food})
        return jsonify({'message': 'food removed from favorite list'}), 200
    
    except Exception as e:
        print('Error removing food:', e)
        return jsonify({'error': 'an error occurred while removing the food'}), 500

# Load and preprocess the data
with open('preprocessed_data.pkl', 'rb') as file:
    df = pickle.load(file)
df['nama'] = df['nama'].str.lower()
food_names = df['nama'].values
categories = df['kategori'].values

# Tokenize food names
tokenizer = Tokenizer()
tokenizer.fit_on_texts(food_names)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(food_names)
padded_sequences = pad_sequences(sequences)

# Load the model and item latent factors
model = keras.models.load_model('recsys_version_1.h5')
item_embedding_model = keras.models.Model(inputs=model.input, outputs=model.layers[0].output)
item_latent_factors = item_embedding_model.predict(padded_sequences)
item_latent_factors = item_latent_factors.reshape((item_latent_factors.shape[0], -1))

# Calculate item-item similarity matrix
item_similarities = np.dot(item_latent_factors, item_latent_factors.T)

@app.route('/recommend/<food_name>')
def recommend_food(food_name):
    # Get recommendations for a specific food
    query_food_index = np.where(food_names == food_name)[0][0]
    query_item_similarities = item_similarities[query_food_index]
    most_similar_indices = np.argsort(query_item_similarities)[-26:-1]  # Get top 25 most similar food indices
    recommended_foods = np.unique(food_names[most_similar_indices])
    
    # list of recommendation dict
    recommendations = []
    for food in recommended_foods:
        food_id = df[df['nama'] == food]['UniqueID'].values[0]
        calories = df[df['nama'] == food]['kalori'].values[0]
        carbs = df[df['nama'] == food]['karbs'].values[0]
        fat = df[df['nama'] == food]['lemak'].values[0]
        protein = df[df['nama'] == food]['protein'].values[0]
        
        recommendation = {
            'food_id': str(food_id),
            'food_name': food,
            'calories': calories,
            'carbs': carbs,
            'fat': fat,
            'protein': protein
        }
        recommendations.append(recommendation)

    # Return recommendations as JSON
    return jsonify(recommendations)

@app.route('/prediction/<seed_text>')
def get_predictions(seed_text):
    # Load the tokenizer from JSON file
    with open('tokenizer.json', 'r') as f:
        tokenizer_json = json.load(f)

    tokenizer = tokenizer_from_json(tokenizer_json)

    # Load the model
    model = load_model('spell_predict_version_1.h5')

    # Define total sentences to predict
    total_sentences = 5
    max_sequence_len = 20

    # Initialize a list to store the predicted sentences
    predicted_sentences = []

    # Loop until the desired number of sentences is reached
    for _ in range(total_sentences):
        # Define total words to predict for each sentence
        next_words = np.random.choice([1, 2, 3])  # Randomly select the number of next words to predict

        # Reset seed text for each sentence
        sentence = seed_text

        # Generate sentence
        for _ in range(next_words):
            # Convert the seed text to a token sequence
            token_list = tokenizer.texts_to_sequences([sentence])[0]

            # Pad the sequence
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

            # Feed to the model and get the probabilities for each index
            probabilities = model.predict(token_list)

            # Pick a random number from [1,2,3]
            choice = np.random.choice(range(1, 11))

            # Sort the probabilities in ascending order
            # and get the random choice from the end of the array
            predicted = np.argsort(probabilities)[0][-choice]

            # Ignore if index is 0 because that is just the padding.
            if predicted != 0:
                # Look up the word associated with the index.
                output_word = tokenizer.index_word[predicted]

                # Combine with the seed text
                sentence += " " + output_word

        # Add the predicted sentence to the list
        predicted_sentences.append(sentence)

    # Create a dictionary to hold the result
    result = {
        "predictions": predicted_sentences
    }

    # Return the result as JSON response
    return jsonify(result)

def load_and_get_recommendations(title):
    # Load the preprocessed data
    with open('preprocessed_data.pkl', 'rb') as file:
        df = pickle.load(file)

    df['kalori'] = df['kalori'].astype(str)
    df['nama'] = df['nama'].str.lower()

    # Converts a collection of raw documents to a matrix of TF-IDF features
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(df['overall'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    df = df.reset_index(drop=True)
    titles = df['nama']  # Defining a new variable title
    indices = pd.Series(df.index, index=df['nama'])  # Defining a new dataframe indices

    # Convert a collection of text documents to a matrix of token counts
    count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    count_matrix = count.fit_transform(df['overall'])

    # Compute cosine similarity between samples in X and Y.
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Define the function that returns 30 most similar movies based on the cosine similarity score
    def get_recommendations(title):
        idx = indices[title]  # Defining a variable with indices
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:31]  # Taking the 30 most similar movies
        movie_indices = [i[0] for i in sim_scores]
        return titles.iloc[movie_indices]  # returns the title based on movie indices

    # Call the get_recommendations function with the provided title
    recommendations = get_recommendations(title)

    # Convert the pandas Series to a list
    recommendations_list = recommendations.tolist()

    # Return the recommendations as a list
    return recommendations_list

@app.route('/recommendations/<title>')
def recommendations(title):
    # Call the load_and_get_recommendations function with the provided title
    recommendations = load_and_get_recommendations(title)

    # Return the recommendations as JSON
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1945)

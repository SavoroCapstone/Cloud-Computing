from flask import Flask, jsonify, request
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials

cred = credentials.Certificate("api.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)

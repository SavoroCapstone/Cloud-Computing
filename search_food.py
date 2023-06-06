from flask import Flask, request, jsonify
from fatsecret import Fatsecret
import random
    
# Set up FatSecret API credentials
consumer_key = '47cd76bdcb1a4a47b1d8fe9b5c619519'
consumer_secret = 'a85a7efd87d7448089fa7fb93a7f175b'
fs = Fatsecret(consumer_key, consumer_secret)
app = Flask(__name__)

# food search on searchbar menu /food-search?query=<food_name>
@app.route('/food-search')
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
@app.route('/food-search')
def search_food():
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

if __name__ == '__main__':
    app.run(debug=True)

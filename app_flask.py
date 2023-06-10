from flask import Flask, request, jsonify
from fatsecret import Fatsecret
import random
    
# Set up FatSecret API credentials
consumer_key = '47cd76bdcb1a4a47b1d8fe9b5c619519'
consumer_secret = 'a85a7efd87d7448089fa7fb93a7f175b'
fs = Fatsecret(consumer_key, consumer_secret)
app = Flask(__name__)

# /food-detail?food_id=<food_id>
@app.route('/food-details')
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
@app.route('/food-search')
def search_food_menu():
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
@app.route('/food-search-menu')
def search_food():
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

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
from fatsecret import Fatsecret
    
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

if __name__ == '__main__':
    app.run(debug=True)

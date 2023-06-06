const express = require("express");
const cors = require("cors");
const app = express();
const bodyParser = require('body-parser');
const admin = require("firebase-admin");
const credentials = require("./api.json");
const path = require("path");
app.use(cors());

admin.initializeApp({
    credential: admin.credential.cert(credentials),
  });

const db = admin.firestore();
app.use(express.json());
app.use(bodyParser.json());
app.use(express.urlencoded({ extended: true }));

// get user data by id
app.get('/documents/:id', async (req, res) => {
    try {
        const docId = req.params.id;
  
        // Retrieve the document from Firestore
        const docRef = db.collection('users').doc(docId);
        const doc = await docRef.get();
  
        // Check if document exists
        if (!doc.exists) {
            return res.status(404).json({ message: 'document not found' });
        }
  
        // Get document data
        const data = doc.data();
  
        return res.status(200).json(data);
    } catch (error) {
        console.error('error retrieving document:', error);
        return res.status(500).json({ message: 'internal server error' });
    }
});
  

// adding favorite food 
app.post('/love', async (req, res) => {
    try {
        const userId = req.body.userId;
        const itemData = req.body.itemData;
  
        const userRef = db.collection('users').doc(userId);
        const userDoc = await userRef.get();
  
        if (!userDoc.exists) {
            // Create user document if doesn't exist
            await userRef.set({ favorite: [itemData] });
        } else {
            // Update user document
            await userRef.update({
                favorite: admin.firestore.FieldValue.arrayUnion(itemData)
            });
        }
  
        res.status(200).send('item added to favorites');
    } catch (error) {
        console.error('error transferring item:', error);
        res.status(500).send('error transferring item');
    }
});
  

// delete fav food in document    
app.delete('/users/:userId/favorite/:foodId', async (req, res) => {
    try {
        const userId = req.params.userId;
        const foodId = req.params.foodId;
  
        // Get user document reference
        const userRef = db.collection('users').doc(userId);
  
        // Get the user document data
        const userDoc = await userRef.get();
        if (!userDoc.exists) {
            res.status(404).json({ error: 'user not found' });
            return;
        }
  
        // Get current favorite food array from the user document
        const favoriteFood = userDoc.data().favorite || [];
  
        // Find the index of the food to be deleted
        const foodIndex = favoriteFood.findIndex(food => food.food_id === foodId);
        console.log(foodId)
        if (foodIndex === -1) {
            res.status(404).json({ error: 'food not found in favorite food list' });
            return;
        }
  
        // Remove food from favorite food array
        favoriteFood.splice(foodIndex, 1);
  
        // Update user document with updated favorite food array
        await userRef.update({ favorite: favoriteFood });
  
        res.status(200).json({ message: 'food removed from favorite list' });
    } catch (error) {
        console.error('error removing food:', error);
        res.status(500).json({ error: 'an error occurred while removing the food' });
    }
});
  
const port = 8085;
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
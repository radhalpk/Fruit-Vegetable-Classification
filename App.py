import streamlit as st
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os

# Load the model
model = load_model('FV.h5')

# Define labels
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 20: 'onion', 21: 'orange',
          22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish',
          29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

# Split into fruits and vegetables
fruits = ['Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']

vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Radish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

# Function to fetch calorie information
def fetch_calories(prediction):
    try:
        search_query = f"calories in {prediction}"
        url = f"https://www.google.com/search?&q={search_query}"
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        
        # Note: Google frequently updates its structure, so you may need to inspect and adjust this
        calorie_info = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calorie_info
    except Exception as e:
        st.error("Unable to fetch calorie information.")
        print(f"Error: {e}")
        return None

# Function to prepare the image for prediction
def prepare_image(img_path):
    try:
        img = load_img(img_path, target_size=(224, 224, 3))
        img = img_to_array(img)
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)
        
        prediction = model.predict(img)
        y_class = np.argmax(prediction, axis=1)
        predicted_label = labels[int(y_class)]
        return predicted_label.capitalize()
    except Exception as e:
        st.error("Error in image processing.")
        print(f"Error: {e}")
        return None

# Main Streamlit function
def run():
    st.title("Nutri AI Image Recognition")
    
    # File uploader for image input
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    
    if img_file is not None:
        # Display uploaded image
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        
        # Save the uploaded image
        save_image_path = os.path.join('upload_images', img_file.name)
        os.makedirs('upload_images', exist_ok=True)
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # Predict and display results
        result = prepare_image(save_image_path)
        if result:
            if result in vegetables:
                st.info(f"**Category**: Vegetables")
            else:
                st.info(f"**Category**: Fruit")
            
            st.success(f"**Predicted**: {result}")
            
            # Fetch and display calorie information
            cal = fetch_calories(result)
            if cal:
                st.warning(f"**{cal} (100 grams)**")

# Run the app
if __name__ == "__main__":
    run()

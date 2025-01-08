import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st

# Set page config
st.set_page_config(page_title="House Price Prediction", layout="wide")

# Load the dataset
data = pd.read_csv('./Housing.csv')

# Prepare features and target for the model
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
          'basement', 'hotwaterheating', 'airconditioning', 'parking',
          'prefarea', 'furnishingstatus']]
Y = data['price']

# Handle categorical columns using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Test the model
predicted_test_prices = model.predict(X_test)
mse = mean_squared_error(Y_test, predicted_test_prices)

# House images, details, and amenities for Home Page
house_images = [
    "https://images.pexels.com/photos/106399/pexels-photo-106399.jpeg",
    "https://postandporch.com/cdn/shop/articles/AdobeStock_209124760.jpg",
    "https://assets.architecturaldesigns.com/plan_assets/358136812/original/144108UPR_Render_1701287518.jpg",
]
house_details = [
    {"name": "Modern Villa", "area": 3000, "price": 5000000, "amenities": ["Gym", "Park", "Pool"]},
    {"name": "Cozy Apartment", "area": 1200, "price": 2000000, "amenities": ["Mall", "School", "Hospital"]},
    {"name": "Luxury Penthouse", "area": 5000, "price": 8000000, "amenities": ["Airport", "Theater", "Stadium"]},
]

# Navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "selected_house" not in st.session_state:
    st.session_state.selected_house = None


def go_to_page(page, house=None):
    st.session_state.page = page
    if house:
        st.session_state.selected_house = house


# Pages
if st.session_state.page == "Home":
    st.title("🏡 House Price Prediction App")
    st.write("Explore various houses and predict their prices.")
    st.write("---")

    # Search functionality
    search_query = st.text_input("Search by house name:")
    filtered_houses = [h for h in house_details if search_query.lower() in h["name"].lower()] if search_query else house_details

    # Filter options
    st.sidebar.subheader("Filters")
    min_area = st.sidebar.slider("Minimum Area (sq.ft)", min_value=0, max_value=5000, value=0)
    max_price = st.sidebar.slider("Maximum Price (₹)", min_value=0, max_value=10000000, value=10000000)
    filtered_houses = [h for h in filtered_houses if h["area"] >= min_area and h["price"] <= max_price]

    # Display house cards
    cols = st.columns(len(filtered_houses))
    for i, col in enumerate(cols):
        with col:
            st.image(house_images[i], use_container_width=True)
            st.subheader(filtered_houses[i]["name"])
            st.write(f"Area: {filtered_houses[i]['area']} sq.ft")
            st.write(f"Price: ₹ {filtered_houses[i]['price']:,}")
            if st.button(f"View Details {i+1}", key=f"view_{i}"):
                go_to_page("House Detail", house=filtered_houses[i])

elif st.session_state.page == "House Detail":
    if st.session_state.selected_house:
        house = st.session_state.selected_house
        st.title(house["name"])
        st.write(f"### Area: {house['area']} sq.ft")
        st.write(f"### Price: ₹ {house['price']:,}")
        st.write("### Nearby Amenities:")
        st.write(", ".join(house["amenities"]))
        st.write("### Virtual Tour:")
        st.video("https://www.youtube.com/watch?v=5qap5aO4i9A")  # Replace with actual video link
        st.write("### Agent Details:")
        st.write("Agent Name: John Doe")
        st.write("Contact: +91-9876543210")
        st.write("### Share Feature:")
        st.button("Share on WhatsApp")
        st.write("---")
        st.button("Back to Home", on_click=lambda: go_to_page("Home"))
    else:
        st.error("No house selected.")
        st.button("Back to Home", on_click=lambda: go_to_page("Home"))

elif st.session_state.page == "Predict":
    st.title("🏠 Predict House Price")
    st.write("### Enter the details of the house:")

    # User inputs
    user_area = st.number_input("Enter the area of the house (in sq.ft):", value=1000)
    user_bedrooms = st.number_input("Enter the number of bedrooms:", value=2, step=1)
    user_bathrooms = st.number_input("Enter the number of bathrooms:", value=1, step=1)
    user_stories = st.number_input("Enter the number of stories:", value=1, step=1)
    user_mainroad = st.radio("Is the house on the main road?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    user_guestroom = st.radio("Does the house have a guest room?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    user_basement = st.radio("Does the house have a basement?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    user_hotwaterheating = st.radio("Does the house have hot water heating?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    user_airconditioning = st.radio("Does the house have air conditioning?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    user_parking = st.number_input("Enter the number of parking spaces:", value=1, step=1)
    user_prefarea = st.radio("Is the house in a preferred area?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    user_furnishingstatus = st.selectbox("Select the furnishing status:", ["semi-furnished", "furnished", "unfurnished"])

    # Convert user inputs into a dataframe
    user_input = pd.DataFrame({
        'area': [user_area],
        'bedrooms': [user_bedrooms],
        'bathrooms': [user_bathrooms],
        'stories': [user_stories],
        'mainroad': [user_mainroad],
        'guestroom': [user_guestroom],
        'basement': [user_basement],
        'hotwaterheating': [user_hotwaterheating],
        'airconditioning': [user_airconditioning],
        'parking': [user_parking],
        'prefarea': [user_prefarea],
        'furnishingstatus': [user_furnishingstatus]
    })

    # Handle categorical data in user input
    user_input = pd.get_dummies(user_input, drop_first=True)
    user_input = user_input.reindex(columns=X.columns, fill_value=0)

    # Predict the house price
    if st.button("Predict Price"):
        predicted_price = model.predict(user_input)
        st.success(f"The predicted price for the given inputs is: ₹ {predicted_price[0]:,.2f}")

    st.write("---")
    st.write(f"### Model Performance:")
    st.write(f"Mean Squared Error on Test Data: {mse:.2f}")

    st.button("Back to Home", on_click=lambda: go_to_page("Home"))

# Navigation buttons
st.sidebar.button("Home", on_click=lambda: go_to_page("Home"))
st.sidebar.button("Predict", on_click=lambda: go_to_page("Predict"))

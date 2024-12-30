import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import hashlib

# Database Functions
def create_connection():
    try:
        conn = sqlite3.connect('users.db')
        return conn
    except sqlite3.Error as e:
        st.error(f"Error connecting to database: {e}")
        return None

conn = create_connection()
c = conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')
    conn.commit()

def add_userdata(username, password):
    try:
        c.execute('INSERT INTO userstable(username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        st.success("User data added successfully")
    except sqlite3.Error as e:
        st.error(f"Error adding user data: {e}")

def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    return c.fetchall()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Prediction Page
def prediction_page():
    if 'data_train' not in st.session_state:
        st.subheader("Upload CSV file for prediction")
        data_file = st.file_uploader("Upload your CSV file", type=["csv"], key="uploaded_file")
        if data_file is not None:
            data_train = pd.read_csv(data_file)
            st.session_state['data_train'] = data_train
            st.session_state['file_uploaded'] = True
            st.write(data_train.head())  # Display the first few rows of the uploaded data
    else:
        data_train = st.session_state['data_train']
        # Display the first few rows of the uploaded data
        st.subheader("First Few Rows of Uploaded Data")
        st.write(data_train.head())

        # Display the shape of the dataset
        st.write(f"Shape of the dataset: {data_train.shape}")

        # Convert columns to numeric
        data_train["views"] = pd.to_numeric(data_train["views"], errors='coerce')
        data_train["comment"] = pd.to_numeric(data_train["comment"], errors='coerce')
        data_train["likes"] = pd.to_numeric(data_train["likes"], errors='coerce')
        data_train["dislikes"] = pd.to_numeric(data_train["dislikes"], errors='coerce')
        data_train["adview"] = pd.to_numeric(data_train["adview"], errors='coerce')
        
        # Encode features like Category, Duration, Vidid
        data_train['duration'] = LabelEncoder().fit_transform(data_train['duration'])
        data_train['vidid'] = LabelEncoder().fit_transform(data_train['vidid'])
        data_train['published'] = LabelEncoder().fit_transform(data_train['published'])

        # Removing rows with the character "F"
        data_train = data_train[data_train.views != 'F']
        data_train = data_train[data_train.likes != 'F']
        data_train = data_train[data_train.dislikes != 'F']
        data_train = data_train[data_train.comment != 'F']

        # Convert views, likes, dislikes, comment columns to numeric, coercing errors to NaN
        data_train["views"] = pd.to_numeric(data_train["views"], errors='coerce')
        data_train["likes"] = pd.to_numeric(data_train["likes"], errors='coerce')
        data_train["dislikes"] = pd.to_numeric(data_train["dislikes"], errors='coerce')
        data_train["comment"] = pd.to_numeric(data_train["comment"], errors='coerce')

        # Drop rows with NaN values
        data_train = data_train.dropna(subset=["views", "likes", "dislikes", "comment"])

        # Remove outliers
        st.subheader("Remove Outliers")
        st.write("Removing videos with views greater than 2000000 as outliers...")
        data_train = data_train[data_train["views"] < 2000000]

        # Display the cleaned data summary
        st.subheader("Cleaned Data Summary")
        st.write(data_train.describe())

        # Assigning each category a number for the Category feature
        category_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
        data_train["category"] = data_train["category"].map(category_mapping)

        # Display the first few rows of the updated DataFrame
        st.subheader("First Few Rows of Updated Data")
        st.write(data_train.head())

        # Visualization
        st.subheader("Visualization")
        
        # Category Histogram
        fig_category_hist, ax_category_hist = plt.subplots()
        ax_category_hist.hist(data_train["category"])
        ax_category_hist.set_title("Category Histogram")
        ax_category_hist.set_xlabel("Category")
        ax_category_hist.set_ylabel("Frequency")
        st.pyplot(fig_category_hist)

        # Adviews Plot
        fig_adviews_plot, ax_adviews_plot = plt.subplots()
        ax_adviews_plot.plot(data_train["adview"])
        ax_adviews_plot.set_title("Adviews Plot")
        ax_adviews_plot.set_xlabel("Index")
        ax_adviews_plot.set_ylabel("Adviews")
        st.pyplot(fig_adviews_plot)

        # Heatmap
        st.subheader("Heatmap of Correlations")
        numeric_columns = data_train.select_dtypes(include=[np.number]).columns.tolist()
        f, ax = plt.subplots(figsize=(10, 8))
        corr = data_train[numeric_columns].corr()
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax, annot=True)
        st.pyplot(f)

        st.write("Columns in dataset:", data_train.columns.tolist())

        required_columns = ['views', 'likes', 'dislikes', 'comment', 'vidid', 'published', 'duration', 'category']
        missing_columns = [col for col in required_columns if col not in data_train.columns]

        if missing_columns:
            st.error(f"Error: The following required columns are missing from the dataset: {', '.join(missing_columns)}")
            st.stop()
       
        # Define the model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42)

        # Model Training
        Y_train = data_train["adview"]
        X_train = data_train.drop(["adview"], axis=1)

        # Save the feature columns used for training
        feature_columns = X_train.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate errors
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Display the errors
        st.subheader("Model Performance")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"Root Mean Squared Error: {rmse}")
        st.write(f"R² Score: {r2}")

        # Select a specific video
        st.subheader("Select a Video for Prediction")
        selected_video = st.selectbox("Select Video", data_train['vidid'].unique())

        # Use the selected video's data for prediction
        selected_video_data = data_train[data_train['vidid'] == selected_video]
        if selected_video_data.shape[0] == 0:
            st.error(f"No data available for the selected video {selected_video}")
            return
        
        selected_video_data = selected_video_data.iloc[0]

        # Ensure the input has the same features as the training data
        X_pred = selected_video_data[feature_columns].values.reshape(1, -1)
        X_pred_scaled = scaler.transform(X_pred)

        # Predict the adviews for the selected video
        adviews_pred = model.predict(X_pred_scaled)[0]
        st.write(f"Predicted Adviews for Selected Video ({selected_video}): {adviews_pred}")

# Title
st.title('YouTube Views Prediction')

# Sidebar for Navigation
st.sidebar.title("Navigation")
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def show_signup():
    st.subheader("Create a new account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type='password')

    if st.button("Signup"):
        if new_user and new_password:
            create_usertable()
            hashed_password = hash_password(new_password)
            add_userdata(new_user, hashed_password)
        else:
            st.warning("Please enter a username and password")

def show_login():
    st.subheader("Login to your account")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        if username and password:
            create_usertable()
            hashed_password = hash_password(password)
            result = login_user(username, hashed_password)
            if result:
                st.success(f"Welcome {username}")
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
            else:
                st.warning("Incorrect Username/Password")
        else:
            st.warning("Please enter a username and password")

def show_logout():
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.success("Logged out successfully")

def show_prediction():
    if st.session_state['logged_in']:
        prediction_page()
    else:
        st.warning("Please login to access this page")

choice = st.sidebar.selectbox("Choose a page", ["Signup", "Login", "Logout", "Prediction"])

if choice == "Signup":
    show_signup()

elif choice == "Login":
    show_login()

elif choice == "Logout":
    show_logout()

elif choice == "Prediction":
    show_prediction()


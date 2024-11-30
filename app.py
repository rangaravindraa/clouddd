from flask import Flask, request, render_template
import pickle
import mysql.connector
from mysql.connector import Error

# Initialize the Flask app
app = Flask(__name__)

# Load the saved models
with open('tfidf_model.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('grid_svm_model.pkl', 'rb') as f:
    grid_svm_model = pickle.load(f)

# MySQL Database Connection
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host='cloud.cpusuq2kwp1i.eu-north-1.rds.amazonaws.com',
            user='admin',         # replace with your MySQL username
            password='12345678', # replace with your MySQL password
            database='cloud'  # replace with your MySQL database name
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print("Error while connecting to MySQL", e)
        return None

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')  # Render the index.html file

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text data from the form input
    text = request.form['text']  # Get the 'text' input field from the form
    
    # Transform the text using the loaded TF-IDF model
    text_tfidf = tfidf.transform([text])
    
    # Make the prediction using the loaded SVM model
    prediction = grid_svm_model.predict(text_tfidf)
    
    # Store the input and output in the database
    store_prediction_in_db(text, prediction[0])
    
    # Render a new template with the prediction result
    return render_template('result.html', prediction=prediction[0], input_text=text)  # Send the result to the result.html page

# Function to store the input and prediction in the database
def store_prediction_in_db(input_text, prediction):
    try:
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            query = "INSERT INTO predictions (input_text, prediction) VALUES (%s, %s)"
            cursor.execute(query, (input_text, prediction))
            connection.commit()
            cursor.close()
            connection.close()
    except Error as e:
        print(f"Error storing data in MySQL: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)

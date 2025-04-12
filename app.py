import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
df = pd.read_csv('Book2.csv')  # Ensure your dataset path is correct
model = pickle.load(open("dt_model.pkl", "rb"))

# Feature columns (input variables)
features = ['Transaction Amount', 'Payment Method', 'Product Category', 'Quantity', 'Customer Age', 
            'Customer Location', 'Device Used', 'Account Age Days', 'Transaction Hour']

# Target variable
target = 'Is Fraudulent'

# Handle categorical columns (Payment Method, Product Category, Customer Location, Device Used) using Label Encoding
label_encoder_payment_method = LabelEncoder()
label_encoder_product_category = LabelEncoder()
label_encoder_customer_location = LabelEncoder()
label_encoder_device_used = LabelEncoder()

df['Payment Method'] = label_encoder_payment_method.fit_transform(df['Payment Method'])
df['Product Category'] = label_encoder_product_category.fit_transform(df['Product Category'])
df['Customer Location'] = label_encoder_customer_location.fit_transform(df['Customer Location'])
df['Device Used'] = label_encoder_device_used.fit_transform(df['Device Used'])

# Handle missing values (NaN values) and infinite values before proceeding with modeling
df.dropna(inplace=True)  # Remove rows with NaN values
print(df.isnull().sum())
print(df.duplicated().sum())

# Prepare the features (X) and target (y)
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model accuracy on the test set
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f'Random Forest Accuracy: {rf_accuracy * 100:.2f}%')

# Save the trained model to disk
with open('rf_model.pkl', 'wb') as rf_model_file:
    pickle.dump(rf_model, rf_model_file)

# Train Decision Tree (DT) model
def train_decision_tree(X_train, y_train, X_test, y_test):
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    dt_y_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_y_pred)

    # Save the trained model to disk
    with open('dt_model.pkl', 'wb') as dt_model_file:
        pickle.dump(dt_model, dt_model_file)

    return dt_accuracy

# Train Convolutional Neural Network (CNN) model
def train_cnn(X_train, y_train, X_test, y_test):
    # Pad sequences to the same length for CNN input
    X_train_padded = pad_sequences(X_train.to_numpy(), padding='post', maxlen=1000)
    X_test_padded = pad_sequences(X_test.to_numpy(), padding='post', maxlen=1000)

    # Build CNN model
    model = Sequential()
    model.add(Conv1D(128, 5, activation='relu', input_shape=(X_train_padded.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Reshape data for CNN
    X_train_padded = X_train_padded.reshape((X_train_padded.shape[0], X_train_padded.shape[1], 1))
    X_test_padded = X_test_padded.reshape((X_test_padded.shape[0], X_test_padded.shape[1], 1))

    # Train the model
    model.fit(X_train_padded, y_train, epochs=10, batch_size=64, verbose=1)

    # Evaluate the model
    _, cnn_accuracy = model.evaluate(X_test_padded, y_test, verbose=0)

    # Save the trained model to disk
    model.save('cnn_model.h5')

    return cnn_accuracy

# Flask app setup
app = Flask(__name__)
app.secret_key = '371023ed2754119d0e5d086d2ae7736b'

# Load the trained models (Random Forest, Decision Tree, CNN models)
def load_rf_model():
    with open('rf_model.pkl', 'rb') as rf_model_file:
        rf_model = pickle.load(rf_model_file)
    return rf_model

def load_dt_model():
    with open('dt_model.pkl', 'rb') as dt_model_file:
        dt_model = pickle.load(dt_model_file)
    return dt_model

def load_cnn_model():
    from tensorflow.keras.models import load_model
    cnn_model = load_model('cnn_model.h5')
    return cnn_model

# Function to safely transform values using label encoder
def safe_transform(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # Handle unknown label (return a default value or use a placeholder)
        return -1  # or any other suitable fallback value

# Flask routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/submit_signup', methods=['POST'])
def submit_signup():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    return render_template('signup_success.html', username=username)

@app.route('/submit_login', methods=['POST'])
def submit_login():
    username = request.form.get('username')
    password = request.form.get('password')
    session['username'] = username
    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model
    if request.method == 'POST':
        # Get the input transaction data from the form
        transaction_amount = float(request.form['Transaction Amount'])
        payment_method = request.form['Payment Method']
        product_category = request.form['Product Category']
        quantity = float(request.form['Quantity'])
        customer_age = float(request.form['Customer Age'])
        customer_location = request.form['Customer Location']
        device_used = request.form['Device Used']
        account_age_days = int(request.form['Account Age Days'])
        transaction_hour = int(request.form['Transaction Hour'])

        # Convert categorical features to numeric using pre-trained label encoders
        payment_method_numeric = safe_transform(label_encoder_payment_method, payment_method)
        product_category_numeric = safe_transform(label_encoder_product_category, product_category)
        customer_location_numeric = safe_transform(label_encoder_customer_location, customer_location)
        device_used_numeric = safe_transform(label_encoder_device_used, device_used)

        # Create a DataFrame for input data
        input_data = pd.DataFrame({
            'Transaction Amount': [transaction_amount],
            'Payment Method': [payment_method_numeric],
            'Product Category': [product_category_numeric],
            'Quantity': [quantity],
            'Customer Age': [customer_age],
            'Customer Location': [customer_location_numeric],
            'Device Used': [device_used_numeric],
            'Account Age Days': [account_age_days],
            'Transaction Hour': [transaction_hour]
        })

        # Load the trained model (Random Forest, Decision Tree, or CNN based on user choice)
        model_type = request.form.get('model_type')

        prediction = None  # Initialize prediction variable

        if model_type == 'rf':
            model = load_rf_model()
            prediction = model.predict(input_data)
        elif model_type == 'dt':
            model = load_dt_model()
            prediction = model.predict(input_data)
        elif model_type == 'cnn':
            model = load_cnn_model()
            input_data_padded = pad_sequences(input_data.to_numpy(), padding='post', maxlen=1000)
            input_data_padded = input_data_padded.reshape((input_data_padded.shape[0], input_data_padded.shape[1], 1))
            prediction = model.predict(input_data_padded)
            # Apply threshold to CNN output
            prediction = 1 if prediction[0] > 0.5 else 0

        # Interpret and return the result
        if prediction is not None:
            if prediction[0] == 0:
                result = 'Fraudulent'
            else:
                result = 'Not Fraudulent'
        else:
            result = "Prediction failed."

        return render_template('result.html', prediction=result)

    return render_template('predict.html')


@app.route('/run_rf')
def run_rf():
    # Simulating Random Forest result here
    return render_template('rf_algorithm.html', rf_accuracy=rf_accuracy)

@app.route('/run_dt')
def run_dt():
    # Train Decision Tree model and get accuracy
    dt_accuracy = train_decision_tree(X_train, y_train, X_test, y_test)
    return render_template('run_dt.html', dt_accuracy=dt_accuracy)

@app.route('/run_cnn')
def run_cnn():
    # Train CNN model and get accuracy
    cnn_accuracy = train_cnn(X_train, y_train, X_test, y_test)
    return render_template('run_cnn.html', cnn_accuracy=cnn_accuracy)

if __name__ == '__main__':
    app.run(debug=True)

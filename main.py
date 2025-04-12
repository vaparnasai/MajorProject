import numpy as np
import pandas as pd
import pickle
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
df = pd.read_csv('Book1.csv')  # Ensure your dataset path is correct
rf_model = pickle.load(open("rf_model.pkl", "rb"))
dt_model = pickle.load(open("dt_model.pkl", "rb"))
cnn_model = load_model('cnn_model.h5')

# Label Encoders for categorical features
label_encoder_payment_method = LabelEncoder()
label_encoder_product_category = LabelEncoder()
label_encoder_customer_location = LabelEncoder()
label_encoder_device_used = LabelEncoder()

# Fit LabelEncoders on data
df['Payment Method'] = label_encoder_payment_method.fit_transform(df['Payment Method'])
df['Product Category'] = label_encoder_product_category.fit_transform(df['Product Category'])
df['Customer Location'] = label_encoder_customer_location.fit_transform(df['Customer Location'])
df['Device Used'] = label_encoder_device_used.fit_transform(df['Device Used'])

# Feature columns (input variables)
features = ['Transaction Amount', 'Payment Method', 'Product Category', 'Quantity', 'Customer Age',
            'Customer Location', 'Device Used', 'Account Age Days', 'Transaction Hour']

# Target variable
target = 'Is Fraudulent'

# Clean input data function
def clean_input_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.clip(upper=1e10)
    df = df.astype(np.float64)
    return df

# Preprocess input data function
def preprocess_input(input_data):
    # Convert categorical data to numeric using pre-trained label encoders
    input_data['Payment Method'] = label_encoder_payment_method.transform([input_data['Payment Method']])
    input_data['Product Category'] = label_encoder_product_category.transform([input_data['Product Category']])
    input_data['Customer Location'] = label_encoder_customer_location.transform([input_data['Customer Location']])
    input_data['Device Used'] = label_encoder_device_used.transform([input_data['Device Used']])

    input_data = pd.DataFrame([input_data])
    input_data = clean_input_data(input_data)
    input_data = pd.get_dummies(input_data, drop_first=True)
    
    return input_data

# Function to predict using the selected model
def predict_fraud(model_type, input_data):
    if model_type == 'rf':
        prediction = rf_model.predict(input_data)
    elif model_type == 'dt':
        prediction = dt_model.predict(input_data)
    elif model_type == 'cnn':
        # Prepare data for CNN model
        input_data = pad_sequences(input_data.to_numpy(), padding='post', maxlen=1000)
        input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
        prediction = cnn_model.predict(input_data)
    
    return 'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent'

# Tkinter GUI setup
def run_gui():
    def on_predict():
        try:
            # Gather input data from entry widgets
            transaction_amount = float(entry_transaction_amount.get())
            payment_method = entry_payment_method.get()
            product_category = entry_product_category.get()
            quantity = float(entry_quantity.get())
            customer_age = float(entry_customer_age.get())
            customer_location = entry_customer_location.get()
            device_used = entry_device_used.get()
            account_age_days = int(entry_account_age_days.get())
            transaction_hour = int(entry_transaction_hour.get())
            model_type = model_type_var.get()  # 'rf', 'dt', or 'cnn'
            
            input_data = {
                'Transaction Amount': transaction_amount,
                'Payment Method': payment_method,
                'Product Category': product_category,
                'Quantity': quantity,
                'Customer Age': customer_age,
                'Customer Location': customer_location,
                'Device Used': device_used,
                'Account Age Days': account_age_days,
                'Transaction Hour': transaction_hour
            }

            # Preprocess input data
            input_data_preprocessed = preprocess_input(input_data)
            
            # Make prediction
            result = predict_fraud(model_type, input_data_preprocessed)
            
            # Show result
            messagebox.showinfo("Prediction Result", f"The transaction is {result}.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    # Create Tkinter window
    window = Tk()
    window.title("Fraud Detection App")

    # Input fields for transaction data
    Label(window, text="Transaction Amount").grid(row=0, column=0)
    entry_transaction_amount = Entry(window)
    entry_transaction_amount.grid(row=0, column=1)

    Label(window, text="Payment Method").grid(row=1, column=0)
    entry_payment_method = Entry(window)
    entry_payment_method.grid(row=1, column=1)

    Label(window, text="Product Category").grid(row=2, column=0)
    entry_product_category = Entry(window)
    entry_product_category.grid(row=2, column=1)

    Label(window, text="Quantity").grid(row=3, column=0)
    entry_quantity = Entry(window)
    entry_quantity.grid(row=3, column=1)

    Label(window, text="Customer Age").grid(row=4, column=0)
    entry_customer_age = Entry(window)
    entry_customer_age.grid(row=4, column=1)

    Label(window, text="Customer Location").grid(row=5, column=0)
    entry_customer_location = Entry(window)
    entry_customer_location.grid(row=5, column=1)

    Label(window, text="Device Used").grid(row=6, column=0)
    entry_device_used = Entry(window)
    entry_device_used.grid(row=6, column=1)

    Label(window, text="Account Age Days").grid(row=7, column=0)
    entry_account_age_days = Entry(window)
    entry_account_age_days.grid(row=7, column=1)

    Label(window, text="Transaction Hour").grid(row=8, column=0)
    entry_transaction_hour = Entry(window)
    entry_transaction_hour.grid(row=8, column=1)

    # Model type selection
    Label(window, text="Select Model Type").grid(row=9, column=0)
    model_type_var = StringVar(value="rf")
    Button(window, text="Random Forest", command=lambda: model_type_var.set('rf')).grid(row=9, column=1)
    Button(window, text="Decision Tree", command=lambda: model_type_var.set('dt')).grid(row=9, column=2)
    Button(window, text="CNN", command=lambda: model_type_var.set('cnn')).grid(row=9, column=3)

    # Predict button
    Button(window, text="Predict", command=on_predict).grid(row=10, column=1)

    window.mainloop()

# Run the GUI
if __name__ == '__main__':
    run_gui()

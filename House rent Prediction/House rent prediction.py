# import the needed modules

import pandas as pd
import numpy as np

# Read the data

data=pd.read_csv("House_Rent_Dataset.csv")

# convert all the  categorical features into numerical features to train the house rent prediction model

data["Area Type"] = data["Area Type"].map({"Super Area": 1, "Carpet Area": 2, "Built Area": 3})
data["City"] = data["City"].map({"Mumbai": 4000, "Chennai": 6000, "Bangalore": 5600, "Hyderabad": 5000, "Delhi": 1100, "Kolkata": 7000})
data["Furnishing Status"] = data["Furnishing Status"].map({"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2})
data["Tenant Preferred"] = data["Tenant Preferred"].map({"Bachelors/Family": 2, "Bachelors": 1, "Family": 3})

# split the data into training and test sets

from sklearn.model_selection import train_test_split
x = np.array(data[["BHK", "Size", "Area Type", "City", "Furnishing Status", "Tenant Preferred", "Bathroom"]])
y = np.array(data[["Rent"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

# train the house rent prediction model using an LSTM(Long Short Term Memory) neural network model

from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=21)

# predicting the rent of a housing property using the trained model with the input being acquired from the user

print("Enter House Details to Predict Rent")
a = int(input("Pin Code of the City: "))
b = int(input("Area Type (Super Area = 1, Carpet Area = 2, Built Area = 3): "))
c = int(input("Size of the House: "))
d = int(input("Number of BHK: "))
e = int(input("Number of bathrooms: "))
f = int(input("Furnishing Status of the House (Unfurnished = 0, Semi-Furnished = 1, Furnished = 2): "))
g = int(input("Tenant Type (Bachelors = 1, Bachelors/Family = 2, Only Family = 3): "))

features = np.array([[a, b, c, d, e, f, g]])
print("Predicted House Price = ", model.predict(features))
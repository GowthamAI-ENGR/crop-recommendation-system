# -*- coding: utf-8 -*-
"""
Crop Recommendation Model Training Script
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
crop = pd.read_csv("Crop_recommendation.csv")

# Map crop labels to numbers
crop_dict = {
    'rice': 1,
    'maize': 2,
    'chickpea': 3,
    'kidneybeans': 4,
    'pigeonpeas': 5,
    'mothbeans': 6,
    'mungbean': 7,
    'blackgram': 8,
    'lentil': 9,
    'pomegranate': 10,
    'banana': 11,
    'mango': 12,
    'grapes': 13,
    'watermelon': 14,
    'muskmelon': 15,
    'apple': 16,
    'orange': 17,
    'papaya': 18,
    'coconut': 19,
    'cotton': 20,
    'jute': 21,
    'coffee': 22,
}
crop['label'] = crop['label'].map(crop_dict)

# Prepare features and target
X = crop.drop('label', axis=1)
y = crop['label']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
mx = MinMaxScaler()
x_train_scaled = mx.fit_transform(x_train)
x_test_scaled = mx.transform(x_test)

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train_scaled)
x_test_scaled = sc.transform(x_test_scaled)

# Train and evaluate models
models = {
    'Logistic Regression': LogisticRegression(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Support Vector Classifier': SVC(),
    'K-Neighbors Classifier': KNeighborsClassifier(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Extra Tree Classifier': ExtraTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Bagging Classifier': BaggingClassifier(),
    'Gradient Boosting Classifier': GradientBoostingClassifier(),
    'Ada Boost Classifier': AdaBoostClassifier()
}

print("Model Accuracies:")
for name, model in models.items():
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    score = accuracy_score(y_test, y_pred)
    print(f"{name}: {score:.4f}")

# Train the final model (Random Forest)
randclf = RandomForestClassifier()
randclf.fit(x_train_scaled, y_train)
y_pred = randclf.predict(x_test_scaled)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Random Forest Accuracy: {final_accuracy:.4f}")

# Recommendation function
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = mx.transform(features)
    sc_features = sc.transform(mx_features)
    prediction = randclf.predict(sc_features)
    return prediction[0]

# Test the recommendation function
N = 90
P = 42
K = 43
temperature = 20.879744
humidity = 82.002744
ph = 6.502985
rainfall = 202.935536

predict = recommendation(N, P, K, temperature, humidity, ph, rainfall)
print(f"\nPrediction for sample input: {predict}")

# Save the model and scalers
pickle.dump(randclf, open('model.pkl', 'wb'))
pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))

print("\nModel and scalers saved successfully.")


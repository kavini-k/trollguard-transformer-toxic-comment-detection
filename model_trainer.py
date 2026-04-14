import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from joblib import dump
import numpy as np

# Load the dataset
try:
    df = pd.read_csv("data/cleaned_data.csv")
    print("Columns found:", df.columns.tolist())
    
    # Check for required columns
    if 'comment_text' not in df.columns:
        raise KeyError("'comment_text' column not found in dataset")
    
    # Define target columns (modify as per your actual columns)
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Verify all target columns exist
    missing_cols = [col for col in target_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing target columns: {missing_cols}")
    
    # Prepare data
    X = df['comment_text']  # Text features
    y = df[target_columns]  # Multilabel targets
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train multilabel model
    base_model = LogisticRegression(max_iter=1000, solver='saga')
    model = MultiOutputClassifier(base_model)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    train_score = model.score(X_train_vec, y_train)
    test_score = model.score(X_test_vec, y_test)
    print(f"Training Accuracy: {train_score:.2f}")
    print(f"Test Accuracy: {test_score:.2f}")
    
    # Save models
    dump(vectorizer, "models/vectorizer.joblib")
    dump(model, "models/toxicity_model.joblib")
    print("Models saved successfully!")
    
except Exception as e:
    print(f"Error: {str(e)}")
    print("\nTroubleshooting steps:")
    print("1. Verify cleaned_data.csv exists in data/ folder")
    print("2. Check column names match your dataset")
    print("3. Ensure all target columns exist in the CSV")
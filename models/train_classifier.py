import sys
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib


def load_data(database_filepath):
    """Load data from database and return X, Y, and category names

    Args:
        database_filepath (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Read df from SQL database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("DRTable", engine)
    # Split df into X and Y
    X = df["message"]
    Y = df.drop(columns=["id", "message", "original", "genre"])
    # Get category names
    category_names = Y.columns.values
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize text and return list of tokens

    Args:
        text (string): String to be tokenized

    Returns:
        _type_: List of tokens
    """

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    tokens = word_tokenize(text)
    # Lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    return tokens


def build_model():
    """
    Build model and return model

    Returns:
        sklearn model: Built model
    """

    # Grid search params
    parameters = {
        "clf__estimator__n_estimators": [10, 20],
        "clf__estimator__min_samples_split": [2, 4]
    }

    # Build model to be used in app
    pipe = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    grid = GridSearchCV(pipe, parameters, cv=2)
    return grid


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model and print classification report

    Args:
        model (sklearn model): sklearn model
        X_test (numpy ndarray): test features
        Y_test (numpy 1darray): test labels
        category_names (list): list of category names

    Returns:
        1d array: predictions
    """
    # Evaluate model
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))
    return Y_pred
    


def save_model(model, model_filepath):
    # Save model to pkl with joblib
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print("Y vals:")
        print(Y.columns.values)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model = model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
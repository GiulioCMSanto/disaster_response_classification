import sys
import re
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from joblib import dump

def remove_outliers(df):
    """
    Removes rows that contains less than 25 characters. These rows
    contains only metacharacters and are not usefull for classification.

    Arguments:
        df: modeling dataframe
    
    Output:
        df_clean: dataframe with outliers removed
    """
    df_clean = df.copy()

    text_length = df_clean['message'].apply(lambda x: len(x))
    outliers = df_clean['message'][text_length[text_length < 25].index].values
    df_clean = df_clean[~df_clean['message'].isin(outliers)]
    
    return df_clean

def load_data(database_filepath):
    """
    Reads the database, converts it into dataframe and performs basic
    outlier removals.

    Arguments:
        database_filepath: the filepath to the database
    
    Output:
        X: features array
        y: multi-target output array
        category_names: targets names

    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql(f"SELECT * FROM {database_filepath.split('/')[1].split('.')[0]}",
                     engine)

    df = remove_outliers(df)

    category_names = list(df.iloc[:,4:].columns)
    X = df['message'].values
    y = df.iloc[:,4:].values

    return X, y, category_names

def tokenize(text):
    """
    Tokenizes an input text data, performing regex, lemmatization and stemming 
    and removing symbols (metacharacters) and stopwords.

    Arguments:
        text: an input string
    
    Output:
        clean_tokens: the tokenized string
    """
    #Remove urls using regex
    url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    repeated_symbols_regex = r"[\?\.\!]+(?=[\?\.\!])"
    
    text = re.sub(url_regex,"urlplaceholder",text)
    text = re.sub(repeated_symbols_regex,'', text)
    
    #Tokenize text
    tokens = word_tokenize(text)
    
    #Lemmatizer and stopwords
    clean_tokens = [WordNetLemmatizer().lemmatize(w.lower().strip())
                    for w in tokens if w not in stopwords.words('english')]
    
    #Stemmer
    clean_tokens = [PorterStemmer().stem(t) for t in clean_tokens]
    
    #Removing Symbols
    symbols_list = ['_','-','?','!','.','@','#','$','%','^','&','*','(',')','[',']','/']
    clean_tokens = [PorterStemmer().stem(t) for t in clean_tokens if t not in symbols_list]
    
    return clean_tokens


def build_model():
    """
    Builds a Machine Learning Pipeline as well as a
    randomized search for parameters that can be used
    for training.
    """
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1),n_jobs=-1))
        ])

    param_distributions = {'clf__estimator__n_estimators': [10],
                        'clf__estimator__criterion':['gini','entropy'],
                        'clf__estimator__max_depth':list(range(1,10))+list(range(10,100,10))+list(range(100,1100,100)),
                        'clf__estimator__min_samples_split': list(range(2,20)),
                        'clf__estimator__min_samples_leaf': list(range(2,20))}

    cv = RandomizedSearchCV(pipeline,
                            param_distributions=param_distributions,
                            n_iter=20,
                            cv=3)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evalautes the model performance printing a classification
    report.

    Arguments:
        model: the trained classifier
        X_test: the testing features
        Y_test: the testing targets
        category_names: the targets names
    """

    y_pred = model.predict(X_test)
    
    for i in range(y_pred.shape[1]):
        print(f"Feature Name: {category_names[i]}")
        print(classification_report(Y_test[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    """
    Saves the final model.

    Argument:
        model: the final trained model
        model_filepath: outpath to save the model
    """
    dump(model, model_filepath) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train,y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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
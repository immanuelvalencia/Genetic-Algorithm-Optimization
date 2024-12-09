import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load training and testing datasets
df_train = pd.read_csv(
    'diabetes_train.csv', 
    names=['gender', 'age', 'hypertension', 'heart_disease', 
           'smoking_history', 'bmi', 'HbA1c_level', 
           'blood_glucose_level', 'diabetes']
)
df_test = pd.read_csv(
    'diabetes_test.csv', 
    names=['gender', 'age', 'hypertension', 'heart_disease', 
           'smoking_history', 'bmi', 'HbA1c_level', 
           'blood_glucose_level', 'diabetes']
)

df_train = df_train.dropna()
df_test = df_test.dropna()

# Function to prepare data for training and testing
def prepare_data(X, selected_features):
    """
    Prepares the data by selecting features, scaling, and splitting into train and test sets.
    """
    X_train = df_train.iloc[:, selected_features].values
    y_train = df_train.iloc[:, 8].values  # Target column

    X_test = df_test.iloc[:, selected_features].values
    y_test = df_test.iloc[:, 8].values  # Target column

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Function for K-Nearest Neighbors
def knn_classifier(X_train, X_test, y_train, y_test):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return -accuracy  # Return negative accuracy for minimization

# Function for Random Forest
def random_forest_classifier(X_train, X_test, y_train, y_test):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return -accuracy  # Return negative accuracy for minimization

# Function for Logistic Regression
def logistic_regression_classifier(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return -accuracy  # Return negative accuracy for minimization

# Function for Support Vector Machine
def svm_classifier(X_train, X_test, y_train, y_test):
    classifier = SVC(kernel='linear', random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return -accuracy  # Return negative accuracy for minimization

# Function for Decision Tree
def decision_tree_classifier(X_train, X_test, y_train, y_test):
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return -accuracy  # Return negative accuracy for minimization

# Objective function for Genetic Algorithm
def f(X):
    """
    Genetic Algorithm Objective Function: 
    Evaluates the classification accuracy of the selected features using a specified algorithm.
    """
    selected_features = np.where(X == 1)[0]
    
    if len(selected_features) == 0:
        return 0  # No features selected, fitness is 0

    # Prepare the data
    X_train, X_test, y_train, y_test = prepare_data(X, selected_features)
    
    # Use one of the classifiers (switch this to test other classifiers)
    # return knn_classifier(X_train, X_test, y_train, y_test)
    # return random_forest_classifier(X_train, X_test, y_train, y_test)
    # return logistic_regression_classifier(X_train, X_test, y_train, y_test)
    # return svm_classifier(X_train, X_test, y_train, y_test)
    return knn_classifier(X_train, X_test, y_train, y_test)

# Parameters for Genetic Algorithm
algorithm_param = {
    'max_num_iteration': 100,
    'population_size': 100,
    'mutation_probability': 0.1,
    'elit_ratio': 0,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': 30
}

# Define variable bounds (binary for feature selection)
varbound = np.array([[0, 1]] * 8)

# Initialize and run the Genetic Algorithm
model = ga(
    function=f, 
    dimension=8, 
    variable_type='int', 
    variable_boundaries=varbound, 
    algorithm_parameters=algorithm_param
)

model.run()

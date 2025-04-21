'''
CSCI425 Final Project
Analyzing Traffic Disruptions in Relation to Cold Temperatures
By Maison Kasprick and Cale Voglewede

csci425_final_project.py
The main file to be executed for the project.
'''

from preprocessing import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def standardization(data:list) -> list:
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # print(f'{X} {y}') # Use to check data separation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Standarize X_train and X_test
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Saving all x and y variables to output
    variables = []
    variables.append(X_train_std)
    variables.append(X_test_std)
    variables.append(y_train)
    variables.append(y_test)
    return variables


def random_forest_results(X_train:list, X_test:list, y_train:list, y_test:list, dataset:str) -> None:
    # print(f'{X_train} {X_test} {y_train} {y_test}') # Use to check match with cold and warm variables
    print(f'{dataset} Dataset')
    print(f'------------')
    forest = RandomForestClassifier().fit(X_train, y_train)
    print('Training Accuracy: %.3f' % forest.score(X_train, y_train))
    print('Testing Accuracy: %.3f' % forest.score(X_test, y_test))
    test_temperatures = None
    if (dataset == "Cold"):
        test_temperatures = np.arange(-30, 33).reshape(-1,1)
    else:
        test_temperatures = np.arange(33, 120).reshape(-1,1)
    
    probabilities = forest.predict_proba(test_temperatures)
    avgs = calculate_average_probabilities(probabilities)
    i = 0
    for avg in avgs:
        print(f'Average Probability of Traffic Disruption with Severity Level of {i+1}: {avg}')
        i += 1
    print()
    
def calculate_average_probabilities(probabilities: np.ndarray) -> list:
    j = 0
    i = 0
    avgs = [0,0,0,0]
    for set_of_probabilities in probabilities:
        for probability in set_of_probabilities:
            avgs[i] += probability
            i += 1
            
        i = 0
        j += 1
    
    avgs = list(map(lambda x: x/j,avgs))
    return avgs
        

def main():
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
    data = preprocessing()
    # print(data) # Use to check data preprocessing
    data_cold = data[0]
    data_warm = data[1]

    # Standardize both the cold and warm datasets
    cold_variables = standardization(data_cold)
    warm_variables = standardization(data_warm)
    # print(f'{cold_variables} {warm_variables}') # Use to check to make sure all variables are there
    X_train_cold, X_test_cold, y_train_cold, y_test_cold = cold_variables
    X_train_warm, X_test_warm, y_train_warm, y_test_warm = warm_variables

    # Pass both cold and warm x and y through linear regression
    random_forest_results(X_train_cold, X_test_cold, y_train_cold, y_test_cold, "Cold")
    random_forest_results(X_train_warm, X_test_warm, y_train_warm, y_test_warm, "Warm")


if __name__ == '__main__':
    main()
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
from sklearn.linear_model import LinearRegression


def standardization(data:list) -> list:
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # print(f'{X} {y}') # Use to check data separation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Standarize X_train and X_test
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    variables = []
    variables.append(X_train_std)
    variables.append(X_test_std)
    variables.append(y_train)
    variables.append(y_test)

    return variables


def regression_results(X_train, X_test, y_train, y_test):
    # print(f'{X_train} {X_test} {y_train} {y_test}') # Use to check match with cold and warm variables
    linear = LinearRegression().fit(X_train, y_train)
    print('Linear Regression Training Accuracy: %.3f' % linear.score(X_train, y_train))
    print('Linear Regression Testing Accuracy: %.3f' % linear.score(X_test, y_test))


def main():
    data = preprocessing()
    # print(data) # Use to check data preprocessing
    data_cold = data[0]
    data_warm = data[1]

    # Standardize both the cold and warm datasets
    cold_variables = standardization(data_cold)
    warm_variables = standardization(data_warm)
    # print(f'{cold_variables} {warm_variables}') # Use to check to make sure all variables are there

    X_train_cold = cold_variables[0]
    X_test_cold = cold_variables[1]
    y_train_cold = cold_variables[2]
    y_test_cold = cold_variables[3]
    
    X_train_warm = warm_variables[0]
    X_test_warm = warm_variables[1]
    y_train_warm = warm_variables[2]
    y_test_warm = warm_variables[3]

    # Pass both cold and warm x and y through linear regression
    regression_results(X_train_cold, X_test_cold, y_train_cold, y_test_cold)
    regression_results(X_train_warm, X_test_warm, y_train_warm, y_test_warm)


if __name__ == '__main__':
    main()
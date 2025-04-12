'''
CSCI425 Final Project
Analyzing Traffic Disruptions in Relation to Cold Temperatures
By Maison Kasprick and Cale Voglewede

csci425_final_project.py
The main file to be executed for the project.
'''

from preprocessing import preprocessing

def main():
    data = preprocessing()
    print(data)
    data_cold = data[0]
    data_warm = data[1]

if __name__ == '__main__':
    main()
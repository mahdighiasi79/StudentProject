import pandas as pd
import numpy as np
import pickle


categorical_values = ["Known As", "Full Name", "Positions Played", "Best Position", "Nationality", "Image Link",
                      "Club Name", "Club Position", "On Loan", "Preferred Foot", "National Team Name",
                      "National Team Image Link", "National Team Position", "National Team Jersey Number",
                      "Attacking Work Rate", "Defensive Work Rate", "Contract Until", "Joined On", "Club Jersey Number"]


def extract_numerical_values():
    df = pd.read_csv("Fifa 23 Players Data.csv")
    columns = df.columns
    numerical_values = []

    for i in range(len(df)):
        numerical_values.append([])
        for j in range(len(columns)):
            if columns[j] not in categorical_values:
                numerical_values[i].append(df.iloc(0)[i][j])

    with open("numerical values.bin", "wb") as file:
        pickle.dump(numerical_values, file)


if __name__ == "__main__":
    #extract_numerical_values()

    with open("numerical values.bin", "rb") as f:
        nv = pickle.load(f)
        f.close()

    print(nv[:2])

    print("hello")


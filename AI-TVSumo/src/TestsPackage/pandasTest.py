import pandas as pd


def main():
    fullDataFrame = pd.read_csv("../../Data/SumoData.csv")
    print(fullDataFrame.head())
    


if __name__ == '__main__':
    main()

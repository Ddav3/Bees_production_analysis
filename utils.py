import pandas as pd

def EDA(dataframe: pd.DataFrame):
    '''
    A simple function used to easily implement the EDA in the input dataframe 
    '''

    print(
        dataframe.info(), 
        "\n##-------------------------------------------##\n",
        dataframe.describe(), 
        "\n##-------------------------------------------##\n",
        dataframe.head(),
        "\n##-------------------------------------------##\n",
        "\nNumber of rows with null/NaN: " + str(dataframe.size - dataframe.dropna().size) + "\n\n"
    )


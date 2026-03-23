import pandas as pd
import seaborn as sns

def EDA(dataframe: pd.DataFrame, head: int = 5):
    '''
    A simple function used to easily implement the EDA in the input dataframe 
    '''

    print(
        "##-------------- INFO ------------------------------##\n")
    dataframe.info()
    print(
        "\n\n##-------------- DESCRIBE --------------------------##\n",
        dataframe.describe(), 
        "\n\n##-------------- HEAD("+str(head) +") ---------------------------##\n",
        dataframe.head(n = head),
        "\n\n##-------------- CORRELATION -----------------------##\n",
        sns.heatmap(dataframe.corr(numeric_only=True), annot=True),
        "\n\n##-------------- NUMBER OF NAN ---------------------##\n",
        "\nNumber of rows with null/NaN: " + str(dataframe.size - dataframe.dropna().size) + "\n\n"
    )
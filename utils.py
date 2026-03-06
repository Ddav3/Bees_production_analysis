import numpy as np
import pandas as pd
import kagglehub as kgh
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os
import matplotlib.pyplot as plt

def load_bees_datasets()-> tuple[pd.DataFrame, pd.DataFrame, dict[pd.DataFrame]]:
    '''
    The function tries to connect to the Kaggle website, in order to load the Datasets used in the project (through Kagglehub).
    Whenever this works correctly, the datasets are put in a variable and a message of successful result is showed. 
    Should this procedure not work for whatever reason, a non-updated version of the Datasets, which is the first version used
    when the code was first finished, is kept in the folder "datasets" and therefore used in their place. 
    The 3 datasets contain:
    - the honey production in US from 1995 to 2021
    - the toxicity levels on bees
    - the observations of weather effects on bees health
    '''
    try:

        ## First Dataset ##

        honey_production_df = kgh.dataset_load(
            KaggleDatasetAdapter.PANDAS, 
            "mohitpoudel/us-honey-production-19952021",
            "US_honey_dataset_updated.csv"
        )
        print(f" honey_production_df {honey_production_df.shape} : Done! \n")


        ## Second Dataset ##

        apistox_df = kgh.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "baharabaz/apistox-dataset", 
            "apistox/dataset_final.csv"
        )
        print(f" apistox_df {apistox_df.shape} : Done! \n")


        ## Third Dataset (collected in a dictionary) later to be merged ##

        #Accessing the subfolder in which the former datasets are stored
        path = kgh.dataset_download("jocelyndumlao/predicting-honeybee-health-from-hive-and-weather")
        subfolder = [elem for elem in os.listdir(path)]
        datasets_folder = os.path.join(path,subfolder[0]) 

        #Creating a dictionary of datasets in order to store them and make them easily accessible for comparison
        bees_health_weather_dict = {}

        for filename in os.listdir(datasets_folder):
            bees_health_weather_dict[filename.replace(".csv", "")] = pd.read_csv(os.path.join(datasets_folder,filename))
            print(f" {filename} {bees_health_weather_dict[filename.replace(".csv", "")].shape} : Done!")

        print("\nDatasets collection completed.")

    except Exception as e: 

        #should the web access to kaggle not work, the local (possibly not updated) version can be used instead 
        print(f"... {e}.\nProcedure stopped.")
        print("-------------------------------\nLocal loading...")


        ## First Dataset ##

        honey_production_df = pd.read_csv("datasets/US_honey_production.csv")
        print(f" honey_production_df {honey_production_df.shape} : Done! \n")


        ## Second Dataset ##

        apistox_df = pd.read_csv("datasets/apistox.csv")
        print(f" apistox_df {apistox_df.shape} : Done! \n")


        ## Third Dataset (collected in a dictionary) later to be merged ##

        bees_health_weather_dict = {}
        for filename in os.listdir("datasets/bees_health_on_weather"):
            bees_health_weather_dict[filename.replace(".csv", "")] = pd.read_csv(os.path.join("datasets/bees_health_on_weather",filename))
            print(f" {filename} {bees_health_weather_dict[filename.replace(".csv", "")].shape} : Done!")

        print("\nSome problem occurred. Local datasets loaded instead.")
    
    return honey_production_df, apistox_df, bees_health_weather_dict


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
        dataframe.corr(numeric_only=True),
        "\n\n##-------------- NUMBER OF NAN ---------------------##\n",
        "\nNumber of rows with null/NaN: " + str(dataframe.size - dataframe.dropna().size) + "\n\n"
    )


def plot_all(dataframe: pd.DataFrame, by: str, x_col: str, y_col: str):
    '''
    Builds the subplots for a group of plots. The dimension is calculated considering the upper limit of the square root of 
    the number of elements of which the second input's column is composed.
    Inputs: 
        dataframe: the Dataframe
        by: the column from which the plots will be subdivided
        x_col: in the Dataframe, the name of the column that will be used in the x axis
        y_col: in the Dataframe, the name of the column containing the data for the y axis
    '''
    
    elements = dataframe[by].unique()
    rows = int(np.ceil(np.sqrt(len(elements)))) 
    cols= rows
    _, axes = plt.subplots(rows, cols, figsize =(20,20))

    for index, element in enumerate(elements):
        plot = axes[int(index/rows),index%cols]
        filtered_part = dataframe[dataframe[by] == element]
        
        plot.plot(filtered_part[x_col], filtered_part[y_col])
        plot.set_title(str(element))
        plot.tick_params(axis ="both", labelsize = 9)
        plot.grid(alpha = 0.3)

    plt.tight_layout()
    plt.show()


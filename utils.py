import numpy as np
import pandas as pd
import kagglehub as kgh
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


def load_bees_datasets()-> tuple[pd.DataFrame, pd.DataFrame, dict[pd.DataFrame]]:
    '''
    The function tries to connect to the Kaggle website, in order to load the Datasets used in the project (through Kagglehub).
    Whenever this works correctly, the datasets are put in a variable and a message of successful result is displayed. 
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

def load_pesticide_usage_datasets(remove_couples: bool = True, remove_kg:bool = True)-> pd.DataFrame:
    '''
    Simply returns the Datasets from datasets folder, regarding the pesticide usage in US during 2019.
    The method can be extended to numerous years, possibly merging the dataframes.
    Being alone an interesting information, the method is left autonomous so that the data can be 
    analyzed at will.
    Inputs:
    -   remove_couples: being indicated in some rows, the result of multiple compounds combined, the faculty to
        delete these rows is given
    -   remove_kg: for some analysis, the kg of pesticide used might not be of interest, therefore, the faculty to
        delete those columns is given; otherwise, a procedure to fill the missing data with a proportion between max/min is done
        and errors regarding the insertion of the max in the min column and viceversa are corrected

    Source: https://water.usgs.gov/nawqa/pnsp/usage/maps/county-level/
    '''
    pesticide_usage_df = pd.read_csv("datasets/EPest_county_estimates_2019.txt", sep= "\t")
    print(f" pesticide_usage_df {pesticide_usage_df.shape} : Done! \n")

    if remove_couples:
        for compound in pesticide_usage_df.COMPOUND.unique():
            if '&' in compound:
                pesticide_usage_df.drop(pesticide_usage_df[pesticide_usage_df.COMPOUND == compound].index, inplace=True)
    if remove_kg:
        pesticide_usage_df.drop(["EPEST_HIGH_KG", "EPEST_LOW_KG"], axis = 1, inplace=True)
    else:
        pesticide_usage_df["EPEST_LOW_KG"].fillna(-1, axis = 0,inplace=True)

        proportion_estimates = []
        for index in pesticide_usage_df.index:
            min = pesticide_usage_df.loc[index, "EPEST_LOW_KG"]
            max = pesticide_usage_df.loc[index, "EPEST_HIGH_KG"]
            if(min != -1):
                if(min>max):
                    pesticide_usage_df.loc[index, "EPEST_LOW_KG"] = max
                    pesticide_usage_df.loc[index, "EPEST_HIGH_KG"] = min
                
                if(max != 0):
                    proportion_estimates.append(min/max)
            
        min_proportion = sum(proportion_estimates)/len(pesticide_usage_df.EPEST_HIGH_KG)

        to_replace = pesticide_usage_df.EPEST_LOW_KG == -1
        pesticide_usage_df.loc[to_replace, "EPEST_LOW_KG"] = pesticide_usage_df.loc[to_replace, "EPEST_HIGH_KG"] * min_proportion

        pesticide_usage_df["MEAN_KG"] =  (pesticide_usage_df["EPEST_LOW_KG"] +  pesticide_usage_df["EPEST_HIGH_KG"])/2
        pesticide_usage_df.drop(["EPEST_HIGH_KG", "EPEST_LOW_KG"], axis = 1, inplace = True)
        
    
    return pesticide_usage_df

def apistox_support_setup(remove_kg : bool = True)-> pd.DataFrame:
    '''
    A stathic method whose job is only to use various resources to obtain information regarding the compounds
    mentioned in Apistox Datasets (ex: CASRN). It connects the dataset regarding the pesticide usage in US with 
    another one that contains the associations of the compounds with their CASRN, other than establishing, for each row,
    the related state, converting the FIPS number into the name string. 

    For an updated comptox dataset, see: https://comptox.epa.gov/dashboard/chemical-lists/PPDB

    Inputs:
    -   remove_kg: used in order to express to the "load_pesticide_usage_datasets" function to delete the kg usage columns or not
    '''
    #obtaining compstox dataset for compound-CASRN association
    comptox_df = pd.read_csv("datasets/Chemical List PPDB-2026-03-07.csv")
    #cleaning 
    comptox_df.columns = comptox_df.columns.str.replace(" ", "_").map(str.upper)
    comptox_df.PREFERRED_NAME = comptox_df.PREFERRED_NAME.map(str.upper)
    comptox_df = comptox_df[["PREFERRED_NAME", "CASRN"]]
    comptox_df.columns = ["COMPOUND", "CAS"]

    #obtaining the data for pesticide_usage, to merge with compstox
    pesticide_usage_df = load_pesticide_usage_datasets(remove_kg=remove_kg)
    pesticide_usage_df = pesticide_usage_df.merge(comptox_df)
    pesticide_usage_df.drop("COUNTY_FIPS_CODE", axis = 1, inplace=True)
    if not remove_kg:
        mean_kg = pesticide_usage_df.groupby(["COMPOUND", "STATE_FIPS_CODE"]).sum()["MEAN_KG"]
        pesticide_usage_df.drop("MEAN_KG", axis = 1, inplace=True)
        pesticide_usage_df.drop_duplicates(inplace=True)
        pesticide_usage_df["MEAN_KG"] = mean_kg.values
    else:
        pesticide_usage_df.drop_duplicates(inplace=True)

    #replacing the FIPS code with the state name  
    pesticide_usage_df.rename(columns={"STATE_FIPS_CODE": "STATE_CODE"}, inplace=True)
    states_map = {
        1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California',
        8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'District of Columbia', 12: 'Florida', 13: 'Georgia',
        15: 'Hawaii', 16: 'Idaho', 17: 'Illinois', 18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 
        21: 'Kentucky', 22: 'Louisiana', 23: 'Maine', 24: 'Maryland', 25: 'Massachusetts', 
        26: 'Michigan', 27: 'Minnesota', 28: 'Mississippi', 29: 'Missouri', 30: 'Montana', 
        31: 'Nebraska', 32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico', 
        36: 'New York', 37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio', 40: 'Oklahoma', 
        41: 'Oregon', 42: 'Pennsylvania', 44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota', 
        47: 'Tennessee', 48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia', 53: 'Washington', 
        54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming'
    }
    pesticide_usage_df["STATE_NAME"] = pesticide_usage_df.STATE_CODE.map(states_map).map(str.upper)
    pesticide_usage_df.drop(columns="STATE_CODE", inplace=True)

    return pesticide_usage_df


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


def plot_all(dataframe: pd.DataFrame, by: str, x: str, y: str):
    '''
    Builds the subplots for a group of plots. The dimension is calculated considering the upper limit of the square root of 
    the number of elements of which the second input's column is composed.
    Inputs: 
    -   dataframe: the Dataframe
    -   by: the column name from which the plots will be subdivided
    -   x: in the Dataframe, the name of the column that will be used in the x axis
    -   y: in the Dataframe, the name of the column containing the data for the y axis
    '''
    
    elements = dataframe[by].unique()
    size = int(np.ceil(np.sqrt(len(elements)))) 
    _, axes = plt.subplots(size, size, figsize =(20,20))

    for index, element in enumerate(elements):
        plot = axes[int(index/size),index%size]
        filtered_part = dataframe[dataframe[by] == element]
        
        plot.plot(filtered_part[x], filtered_part[y])
        plot.set_title(str(element))
        plot.tick_params(axis ="both", labelsize = 9)
        plot.grid(alpha = 0.3)

    plt.tight_layout()
    plt.show()

def bar_all(dataframe: pd.DataFrame, xlabels: list[str], cols: list[str]):
    '''
    Creates a bar of all the rows of a Dataframe passed in input. The size is computed as the sqrt of the number of rows.
    Adviced for the labels to be the index of the Dataframe.
    Inputs:
    -   dataframe: the Dataframe from which the data and the indexes for the plot structure are gathered.
    -   xlabels: a list of string with the same length as the number of cols that will be used as labels
    -   cols: the names of 
    '''
    if(len(xlabels) != len(cols)):
        print("'xlabes' and 'cols' args must have he same length")
        return
    indexes = dataframe.index
    size = int(np.ceil(np.sqrt(len(indexes)))) 
    _, axes = plt.subplots(size, size, figsize =(21,21))

    for index, element in enumerate(indexes):
        bar = axes[int(index/size),index%size]
        bar.bar(xlabels,  dataframe[cols].loc[element].values.flatten())
        bar.set_title(str(element))
        bar.tick_params(axis ="both", labelsize = 9)
        bar.grid(alpha =0.3)

    plt.tight_layout()
    plt.show()

def random_forest(X_data: pd.DataFrame, y_target: pd.Series, n_estimators:int = 500, train_size:float = 0.8, random_state:int = None)-> tuple[pd.DataFrame, float]:
    '''
    The function executes random_forest on the given dataset, giving weights to the feature that establish the target in input. 
    The result is returned in a Dataframe form that associates the values to their feature, together with the accuracy score totalized.
    Inputs:
    -   X_data: the data, in dataframe form, to which the weights are to be associated. These data will alreayd undergo the "get_dummies"
        procedure, so you shouldn't do it.
    -   y_target: the target, in series form, that is obtained using the X_data.
    -   n_estimators: the number of estimators to use for Random Forest
    -   train_size: the proportion of data to use for training. The rest is used for prediction
    -   random_state: a random state that sets the seed for reproducibility
    '''
    X = pd.get_dummies(X_data)
    target = y_target
    random_forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(X,target, train_size=train_size, random_state=random_state)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    score = accuracy_score(y_true=y_test, y_pred=y_pred)

    print("Score: ", score)

    return pd.DataFrame({"Features": X.columns,
                         "Weight %": random_forest.feature_importances_}
                         ).sort_values(by="Weight %", ascending=False).reset_index(drop=True), score
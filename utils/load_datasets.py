import pandas as pd
import kagglehub as kgh
from kagglehub import KaggleDatasetAdapter
import os

def load_base_datasets(load_honey_production: bool = True, load_apistox: bool = True, load_weather_effects: bool = True)-> tuple[pd.DataFrame, pd.DataFrame, dict[pd.DataFrame]]:
    '''
    Tries to connect to the Kaggle website, in order to load the Datasets used in the project (through Kagglehub).
    Whenever this works correctly, the datasets are put in a variable and a message of successful result is displayed. 
    Should this procedure not work for whatever reason, a non-updated version of the Datasets, which dates back to the first version 
    used when the code was finalised, is kept in the folder "datasets" and therefore used in their place. 
    -   Firstly, a dataset regarding honey production in US from 1995 to 2021 is loaded;
    -   the same procedure is followed for a dataset on the toxicity level of chemical compounds on bees;
    -   finally, a folder with info about the observations of weather effects on bees health is put in a vector. The datasets are
        not immediately merged since it may be better to make possible to analyse each of them separately, if desired.

    In any case, the operations are done if and only if the boolean argument related to the dataframe is set True; otherwise, only 
    the datasets whose argument is true will be loaded, by creating a tuple with only the loaded datasets. (By default, they are 
    all loaded).

    See README for datasets source.
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
    
    if "Unnamed: 0" in honey_production_df.columns:
        honey_production_df.drop(columns=["Unnamed: 0"], inplace=True)

    return honey_production_df, apistox_df, bees_health_weather_dict

def load_pesticide_usage_dataset(year: int = 2019, remove_couples: bool = True, remove_kg:bool = True)-> pd.DataFrame:
    '''
    Returns the pesticide usage in US for given year.
    The method can be extended to multiple years, possibly merging the dataframes.
    Having interesting information on its own, the method is left autonomous so that the data can be analyzed at will.
    Inputs:
    -   year: the year to which the data are related;
    -   remove_couples: the result of multiple compounds combined is showd in some rows, so it allows to delete these rows;
    -   remove_kg: for some analysis, the kg of pesticide used might not be of interest: therefore, if True,
        deletes those columns; otherwise, a procedure to fill the missing data with a proportion between max/min is done
        and errors regarding the insertion of the max in the min column and viceversa are corrected.

    Source: https://water.usgs.gov/nawqa/pnsp/usage/maps/county-level/
    '''
    #verifying the validity of input and loading the dataset
    if year < 1992 or year > 2019:
        print("Error. Years available for pesticide usage: 1992-2019.")
        return
    pesticide_usage_df = pd.read_csv(f"datasets/epest_datasets/EPest_county_estimates_{year}.txt", sep= "\t")
    print(f" pesticide_usage_df {pesticide_usage_df.shape} - year {year}: Done! \n")

    if remove_couples:
        for compound in pesticide_usage_df.COMPOUND.unique():
            if '&' in compound:
                pesticide_usage_df.drop(pesticide_usage_df[pesticide_usage_df.COMPOUND == compound].index, inplace=True)
    if remove_kg:
        pesticide_usage_df.drop(["EPEST_HIGH_KG", "EPEST_LOW_KG"], axis = 1, inplace=True)
    else:
        #procedure for filling the missing cells with a min proportional estimate
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

        #replacing_values
        to_replace = pesticide_usage_df.EPEST_LOW_KG == -1
        pesticide_usage_df.loc[to_replace, "EPEST_LOW_KG"] = pesticide_usage_df.loc[to_replace, "EPEST_HIGH_KG"] * min_proportion
        #overwriting max and min estimate columns with mean 
        pesticide_usage_df["MEAN_KG"] =  (pesticide_usage_df["EPEST_LOW_KG"] +  pesticide_usage_df["EPEST_HIGH_KG"])/2
        pesticide_usage_df.drop(["EPEST_HIGH_KG", "EPEST_LOW_KG"], axis = 1, inplace = True)
    
    return pesticide_usage_df

def load_apistox_support_dataframe(year: int = 2019, remove_kg : bool = True)-> pd.DataFrame:
    '''
    Connects the dataset on pesticide usage in US with another one (comptox), that contains the associations compounds-CASRN, 
    other than linking the info to the respective state, converting its FIPS number into the string name of the state. 

    For an updated comptox dataset, see: https://comptox.epa.gov/dashboard/chemical-lists/PPDB

    Inputs:
    -   year: the year to which the data are related to (most recent by default). Used to call "load_pesticide_usage_datasets" 
        function
    -   remove_kg: used in order to express to the "load_pesticide_usage_datasets" function to delete the kg usage columns or not
    '''
    #obtaining the data for pesticide_usage
    pesticide_usage_df = load_pesticide_usage_dataset(year = year, remove_kg=remove_kg)

    #obtaining compstox dataset for compound-CASRN association
    comptox_df = pd.read_csv("datasets/Chemical List PPDB-2026-03-07.csv")
    #cleaning 
    comptox_df.columns = comptox_df.columns.str.replace(" ", "_").map(str.upper)
    comptox_df.PREFERRED_NAME = comptox_df.PREFERRED_NAME.map(str.upper)
    comptox_df = comptox_df[["PREFERRED_NAME", "CASRN"]]
    comptox_df.columns = ["COMPOUND", "CAS"]

    #merging the datasets
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

def load_complete_inspections_on_weather_df()-> pd.DataFrame:
    '''
    Operates on the third dataset loaded by "load_base_datasets" function, which is put in a vector, so that it's possible to
    analyze the six datasets singularly if desired. 
    First, the datasets related to apiary information and inspections on the hives are merged (apiary part) and the same goes for 
    those regarding the weather observations and stations (weather part). After the names of the columns later to be involved in 
    the merge are fixed, together with the type of their elements, the final dataset is built and the necessary columns' 
    type is fixed.
    '''
    inspections_on_weather_dict = load_base_datasets()[2]

    #building apiary part
    apiary_part = inspections_on_weather_dict["Apiary_Information"].merge(inspections_on_weather_dict["Hive_Information"]).merge(inspections_on_weather_dict["HCC_Inspections"])
    apiary_part.rename(columns={"InsptDate": "Date", }, inplace=True)
    apiary_part.Date = pd.to_datetime(apiary_part.Date, format="%Y-%m-%d")

    #building weather part
    weather_part = inspections_on_weather_dict["Weather_Stations"].merge(inspections_on_weather_dict["Hourly_Weather"]).merge(inspections_on_weather_dict["Weather_Observations"])
    weather_part.rename(columns={"Station_City": "City", }, inplace=True)
    weather_part.Date = pd.to_datetime(weather_part.Date, format="%m/%d/%Y")

    #building final dataset
    inspections_on_weather_df = weather_part.merge(apiary_part, on="Date")
    inspections_on_weather_df.dropna(axis= 0, inplace=True)
    inspections_on_weather_df.columns = inspections_on_weather_df.columns.map(str.upper)
    inspections_on_weather_df.rename(columns = {"INPSECTIONID" : "INSPECTIONID"}, inplace=True)
    inspections_on_weather_df.HEALTHY = inspections_on_weather_df.HEALTHY.map(lambda x : 1 if x == "Yes" else 0)

    #fixing names, types and adding columns for grouping
    inspections_on_weather_df.STATE = inspections_on_weather_df.STATE.map(lambda x: "NORTHCAROLINA" if x == "NC" else "UTAH")
    inspections_on_weather_df[["TEMPERATURE", "HUMIDITY", "DEW_POINT", "WIND_SPEED", "WIND_GUST", "PRESSURE", "PRECIP"]] = inspections_on_weather_df[["TEMPERATURE", "HUMIDITY", "DEW_POINT", "WIND_SPEED", "WIND_GUST", "PRESSURE", "PRECIP"]].apply(lambda x: pd.to_numeric(x.str.strip()) if x.dtype == "object" else pd.to_numeric(x))
    inspections_on_weather_df["YEAR"] = inspections_on_weather_df.DATE.dt.year
    inspections_on_weather_df["MONTH"] = inspections_on_weather_df.DATE.dt.month
    return inspections_on_weather_df

def load_varroa_detection_dataset(load_normalized_data = False)-> pd.DataFrame:
    '''
    Tries to connect to the Kaggle website, in order to load the on Varroa Destructor detection, which is
    used in the project (through Kagglehub).
    The 2 available datasets are actually almost identical: 
    -   hive_monitoring_dataset contains the data as they were gathered
    -   data_Varroa_Detection has the same information, but normalized in order to have an easier analysis

    Input:
    -   load_normalized_data: if True, loads "data_Varroa_Detection" dataset containing normalized data, the other one if False
    '''

    if not load_normalized_data:
        ## Loading monitoring dataset ##
        try:

            varroa_detection_df = kgh.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                "anaisabelcaicedoc/varroa-detection-with-discrete-variables",
                "hive_monitoring_dataset.csv"
            )
        except Exception as e:
            #should the web access to kaggle not work, the local (possibly not updated) version can be used instead 
            print(f"... {e}.\nProcedure stopped.")
            print("-------------------------------\nLocal loading...")   

            varroa_detection_df = pd.read_csv("datasets/varroa_destructor/hive_monitoring_dataset.csv")
            print(f" Varroa Destructor Detection Dataset {varroa_detection_df.shape} : Done! \n")    
        
    else:  
        ## Loading normalized dataset ##
        try:
            varroa_detection_df = kgh.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                "anaisabelcaicedoc/varroa-detection-with-discrete-variables",
                "data_Varroa_Detection.csv"
            )
        except Exception as e:
            #should the web access to kaggle not work, the local (possibly not updated) version can be used instead 
            print(f"... {e}.\nProcedure stopped.")
            print("-------------------------------\nLocal loading...")   

            varroa_detection_df = pd.read_csv("datasets/varroa_destructor/data_Varroa_Detection.csv")    
            print(f" Varroa Destructor Detection Dataset (normalized) {varroa_detection_df.shape} : Done! \n")    

    #fixing names fo the columns
    en_cols_names = ["AVG_TEMPERATURE", "AVG_HUMIDITY", "AVG_CO2", "AVG_TVOC", "WEIGHTED_SUM", "PREDICTED_ALERT"]
    varroa_detection_df.columns = en_cols_names
    varroa_detection_df.PREDICTED_ALERT = varroa_detection_df.PREDICTED_ALERT.map(lambda x : 0 if x == "Bajo" else (1 if x == "Medio" else (2 if x == "Alto" else 3)))

    return varroa_detection_df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_all(dataframe: pd.DataFrame, by: str, x: str, y: str):
    '''
    Builds the subplots for a group of plots. The dimension is calculated considering the upper limit of the square root of 
    the number of elements of which the second input's column is composed.
    Inputs: 
    -   dataframe: the Dataframe
    -   by: the column name by which the plots will be subdivided
    -   x: in the Dataframe, the name of the column that will be used in the x axis
    -   y: in the Dataframe, the name of the column containing the data for the y axis
    '''
    
    #setting up variables
    elements = dataframe[by].unique()
    size = int(np.ceil(np.sqrt(len(elements)))) 
    _, axes = plt.subplots(size, size, figsize =(20,20))

    # making plots 
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
    -   xlabels: a list of string, with the same length as the number of columns, that will be used as labels
    -   cols: the names of columns in the Dataframe containing the data for bars height
    '''
    if(len(xlabels) != len(cols)):
        print("'xlabes' and 'cols' args must have he same length")
        return
    
    #setting up data
    indexes = dataframe.index
    size = int(np.ceil(np.sqrt(len(indexes)))) 
    _, axes = plt.subplots(size, size, figsize =(21,21))

    # making bars
    for index, element in enumerate(indexes):
        bar = axes[int(index/size),index%size]
        bar.bar(xlabels,  dataframe[cols].loc[element].values.flatten())
        bar.set_title(str(element))
        bar.tick_params(axis ="both", labelsize = 9)
        bar.grid(alpha =0.3)

    plt.tight_layout()
    plt.show()
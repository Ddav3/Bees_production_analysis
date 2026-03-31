import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_multiple(x_axis: pd.Series | list, y_cols: pd.DataFrame|pd.Series, xlabel: str = "", ylabel: str = "Values") -> go.Scatter:
    '''
    Creates a graphical representation in plotly that can also contain more than a single plot. It is required that the data 
    are plotted with respect to the same data used in x_axis.
    The method only uses plotly, but it can be easily extended to other libraries with a variable "method"(0 = plotly, 1 = pyplot,...)
    Inputs:
    -   w
    '''
    figure = go.Figure()
    if type(y_cols) == pd.DataFrame:
        for col in y_cols.columns:
                figure.add_trace(go.Scatter(
                    x=x_axis,
                    y=y_cols[col],
                    mode="lines+markers",
                    line=dict(width=2),
                    marker=dict(size=7),
                    name=col,
                    hovertemplate=f'<b>{col}</b><br>Value: %{{y}}<extra></extra>'))  
    else:
        figure.add_trace(go.Scatter(
                x=x_axis,
                y=y_cols,
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=7),
                name=y_cols.name,
                hovertemplate=f'<br>Value: %{{y}}<extra></extra>'))  
         
            
    figure.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        hovermode='x unified',
        width=1000, 
        height=500,
        plot_bgcolor='black',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=0.99, xanchor="right", x=0.99))
    figure.update_xaxes(showgrid=True, gridcolor='lightgray')
    figure.update_yaxes(showgrid=True, gridcolor='lightgray')
    return figure


def plot_all(dataframe: pd.DataFrame, by: str, x: str, y: str) -> plt.Figure:
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
    figure, axes = plt.subplots(size, size, figsize =(20,20))

    # making plots 
    for index, element in enumerate(elements):
        plot = axes[int(index/size),index%size]
        filtered_part = dataframe[dataframe[by] == element]
        
        plot.plot(filtered_part[x], filtered_part[y])
        plot.set_title(str(element))
        plot.tick_params(axis ="both", labelsize = 9)
        plot.grid(alpha = 0.3)

    plt.tight_layout()
    plt.close("all")
    return figure

def bar_single_ys_interactive(x_axis: list[str], y_cols: list[float|int], xlabel: str = "", ylabel: str = "", title: str = "")-> go.Figure:
    '''
    Returns an interactive graph object Bar, that for each element named in x_axis, plots its height from y_cols. Note that x_axis
    and y_cols must have the same length. 
    Inputs:
    -   x_axis: the list of names for the bars
    -   y_cols: the values for each element in x_axis
    -   xlabel: the etiquette to eventually put along the x axis 
    -   ylabel: the etiquette to eventually put along the y axis
    -   title: the title to put above the plot, if present
    '''
    if len(x_axis) != len(y_cols):
        print("the list of names for axis x and the list of values for cols must have the same length")
        return 
    figure = go.Figure()

    figure.add_trace(go.Bar(
        x=x_axis,
        y=y_cols,
        hovertemplate='<b>%{x}</b><br>Failures: %{y:,}<extra></extra>'
    ))

    figure.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        showlegend=False,
        height=500,
        plot_bgcolor='black'
    )
    return figure

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
    figure, axes = plt.subplots(size, size, figsize =(21,21))

    # making bars
    for index, element in enumerate(indexes):
        bar = axes[int(index/size),index%size]
        bar.bar(xlabels,  dataframe[cols].loc[element].values.flatten())
        bar.set_title(str(element))
        bar.tick_params(axis ="both", labelsize = 9)
        bar.grid(alpha =0.3)

    plt.tight_layout()
    plt.close("all")
    return figure

def heatmap_cols_interactive(df: pd.DataFrame, compare_cols: bool = False, title: str = "")-> go.Figure:
    '''
    Returns an interactive version of the correlation matrix, as a graph_object Heatmap. It shows the title for the Heatmap, if given.
    If you wish to evaluate the correlation among columns, set compare_cols to True, otherwise, the correlation between 
    cols and rows is made. 
    '''
    if compare_cols:
        df = df.corr()
    
    figure = go.Figure(data=go.Heatmap(
        x=df.columns.tolist(),        
        y=df.index.tolist(),        
        z=df.values,        
        hovertemplate='<b>%{x} ~ %{y}</b><br>Corr: %{z:.2f}<extra></extra>'
    ))

    figure.update_layout(
        title=title,
        height = 600,
        plot_bgcolor='black'
    )
    return figure
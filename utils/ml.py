import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def linear_regression_and_plot(x_array: list, y_array:list, train_size: float = 0.7, random_state = None, 
                               pred_x_array: list = [], show_pred_results: bool = False, interactive: bool = False,
                               plot_train_test = True, xlabel: str = "", ylabel: str = "") -> go.Figure | plt.Figure:
    '''
    Analyses the trend represented by the arrays in input. Computes the process of linear regression and returns the plots of the results.
    The linear regression's score is printed.
    Inputs:
    -   x_array: the array used along the x axis
    -   pred_x_array: the eventual array to use for a prediction along the x_axis
    -   y_array: the of data that correspond to the elements of x_array
    -   train_size: the size of the training; the test size is computed consequently
    -   random_state: the random state used for replicability
    -   pred_x_array: the list (empty by default) on which it is wished to do predictions after fitting the model. 
        If left empty, no prediction is done
    -   plot_train_test: if True, allows to plot train and test results
    -   show_pred_results: if True, shows the y_pred obtained with the model
    -   interactive: if True, plots the data in an interactive way, with plotly; in a static way, with pyplot, 
        otherwise
    -   xlabel: the string to be put as label along x axis
    -   ylabel: the string to be put as label along y axis
    '''
    #preparing the model   
    X_train, X_test, y_train, y_test = train_test_split(x_array.reshape(-1,1), 
                                                y_array, 
                                                train_size=train_size,
                                                random_state=random_state)
    model= LinearRegression()
    model.fit(X = X_train, y = y_train)
    print(f"Score: {model.score(X_test, y_test)}")

    #doing predictions
    if len(pred_x_array) > 0:
        y_pred = model.predict(pred_x_array.reshape(-1, 1))    
        if show_pred_results:
            print(f" Results of prediction: {y_pred}")

    #plotting the results, if there are data for it
    if plot_train_test:

        #with pyplot
        if not interactive:
            plt.figure(figsize=(9, 5))
            if len(pred_x_array) > 0:
                plt.plot(pred_x_array, y_pred, color="red", label="Prediction")
            plt.scatter(X_train, y_train, color="grey", s=40, alpha=0.7, label="Train")
            plt.scatter(X_test, y_test, color="green", s=60, alpha=0.8, label="Test")
            plt.xlabel(xlabel, fontsize = 11)
            plt.ylabel(ylabel, fontsize = 11)
            plt.title("Trend with Prediction" if len(pred_x_array) > 0 else "Trend")
            plt.legend()
            plt.grid(True, alpha=0.35)
            figure = plt.gcf()
            plt.close()
            return figure
        
        #with plotly
        figure = go.Figure()
        #train and test scattered
        figure.add_trace(go.Scatter(x = X_train.flatten(), y = y_train, mode = "markers",
                                 marker= dict(color = "lightgrey", size = 8, opacity = 0.7),
                                 name = "Train",  # tooltip
                                 hovertemplate='<b>Real Value</b><br>X: %{x:.2f}<br>Y: %{y:.2f}'))
        figure.add_trace(go.Scatter(x=X_test.flatten(), y=y_test, mode='markers',
                                 marker=dict(color='green', size=10, opacity=0.8), name= "Test",
                                 hovertemplate='<b>Real Value</b><br>X: %{x:.2f}<br>Y: %{y:.2f}'))
        #adding the line for prediction
        if len(pred_x_array) > 0:
            figure.add_trace(go.Scatter(x = pred_x_array, y = y_pred, mode = "lines",
                                        marker= dict(color = "red", size = 6),name = "Prediction", 
                                        hovertemplate='<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'))

        figure.update_layout(title="Trend with Prediction" if len(pred_x_array)>0 else "Trend", 
                             xaxis_title=xlabel, 
                             yaxis_title=ylabel,
                             hovermode="closest", 
                             width=1000, height=500, plot_bgcolor='black', 
                             legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.5))
        figure.update_xaxes(showgrid=True, gridcolor='lightgray')
        figure.update_yaxes(showgrid=True, gridcolor='lightgray')
        
        return figure
        

def random_forest(X_data: pd.DataFrame, y_target: pd.Series, n_estimators:int = 500, train_size:float = 0.8, random_state:int = None)-> tuple[pd.DataFrame, float]:
    '''
    Executes random_forest on the given dataset, giving weights to the feature that establish the target in input. 
    The result is returned in a Dataframe form that associates the weights to their feature, together with the accuracy score totalized.
    Inputs:
    -   X_data: the data, in dataframe form, to which the weights are to be associated. These data will undergo 
        the "get_dummies" method
    -   y_target: the target, in series form, that is obtained using the X_data.
    -   n_estimators: the number of estimators to use for Random Forest
    -   train_size: the proportion of data to use for training. The rest is used for testing
    -   random_state: a random state that sets the seed for reproducibility
    '''
    #setting up data
    X = pd.get_dummies(X_data)
    target = y_target
    random_forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    #doing predictions and printing the score
    X_train, X_test, y_train, y_test = train_test_split(X,target, train_size=train_size, random_state=random_state)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    score = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Score: ", score)

    return pd.DataFrame({"Features": X.columns,
                         "Weight %": random_forest.feature_importances_}
                         ).sort_values(by="Weight %", ascending=False).reset_index(drop=True), score

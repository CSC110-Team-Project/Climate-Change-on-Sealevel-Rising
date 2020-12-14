"""CSC110 Final Project 2020: Visualization.py

Copyright and Usage Information
===============================
This file is part of the CSC110 final project: Data Analysis on Rising Sea Level,
developed by Charlie Guo, Owen Zhang, Terry Tu, Vim Du.
This file is provided solely for the course evaluation purposes of CSC110 at University of Toronto St. George campus.
All forms of distribution of this code, whether as given or with any changes, are strictly prohibited.
The code may have referred to sources beyond the course materials, which are all cited properly in project report.
For more information on copyright for this project, please contact any of the group members.

This file is Copyright (c) 2020 Charlie Guo, Owen Zhang, Terry Tu and Vim Du.
"""
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import numpy as np
import csv
from typing import Tuple
# Read all the dataset
# The csv file should be the output of our prediction, use existing ones as sample for now


# Part 1: 3D Scatter Plot
def plot_3d_scatter_plot(dataset: object, x: str, y: str, z: str, color: str) -> object:
    """Given the values of x, y, z axis, plot the corresponding
    3D Scatter Plot"""
    fig = px.line_3d(dataset, x=x, y=y, z=z, color=color)
    return fig


# Part 2: Comparison
def plot_comparison_graph(pred: np.array, actual: np.array, x: str, y: str, model: str) -> object:
    """Given predicted and actual data and the values of x,y axis, plot the graphs of two data sets
     to present a comparison"""
    fig = go.Figure()
    nums = [i for i in range(len(pred))]
    x_1, x_2 = nums, nums
    y_1, y_2 = pred, actual
    fig.add_trace(go.Scatter(x=x_1, y=y_1, mode='lines', name='Prediction'))
    fig.add_trace(go.Scatter(x=x_2, y=y_2, mode='lines', name='Actual'))
    fig.update_layout(
        title="Predicted VS Actual for Model:"+model,
        xaxis_title=x,
        yaxis_title=y,
        yaxis_range=[-100, 100],
        xaxis=dict(range=[nums[0], nums[-1]], autorange="reversed")
    )

    return fig


# Part 3: Animated graph
def plot_animated_graph(ds_1: object, ds_2: object, x: str, y: str, duration: int) -> object:
    """Given three datasets, the values of x, y axis, the duration in unit of months
     plot the graphs of three datasets that is animated with the change of independent variable."""

    frame_list = []
    x_1, x_2 = ds_1[x], ds_2[x]
    y_1, y_2 = ds_1[y], ds_2[y]
    s_year, s_month = calculate_start_date(2013, 12, duration)
    year, month = calculate_end_date(2013, 12, duration)
    # build up the list of frames
    for i in range(len(x_1)):
        current_frame = go.Frame(data=[go.Scatter(x=x_1[0:i], y=y_1[0:i], mode='lines', name='CNN'),
                                       go.Scatter(x=x_2[0:i], y=y_2[0:i], mode='lines', name='Actual')])
        frame_list.append(current_frame)

    fig = go.Figure(
        data=[go.Scatter(x=x_1[0:1], y=y_1[0:1], mode='lines', name='CNN'),
              go.Scatter(x=x_2[0:1], y=y_2[0:1], mode='lines', name='Actual')],
        layout=go.Layout(
            xaxis=dict(range=[str(s_year) + '-' + str(s_month) +
                              '-01', str(year)+'-'+str(month)+'-01'], autorange=False),
            yaxis=dict(range=[-100, 100], autorange=False),
            title='Animation of GMSL variations predicted by CNN Model',
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Start",
                              method="animate",
                              args=[None, {"frame": {"duration": 200,    # change duration to control speed
                                                     "redraw": False},
                                           "fromcurrent": True,
                                           "transition": {"duration": 0}}])])]
        ),
        frames=frame_list
    )
    return fig


# helper functions to make the output of models usable for graphing functions
def merge_date(dataset: object) -> object:
    """Process the dataset and merge the Year and Month
    into one column: Date"""
    temp = dataset
    temp['Date'] = pd.to_datetime(temp[['Year', 'Month']].assign(DAY=1))
    return temp


def write_file(file: str, start_year: int, start_month: int, duration: int, output: object) -> None:
    """Write the predicted data into a csv file."""
    with open(file, mode='w') as f:
        columns = ['Year', 'Month', 'GMSL']
        writter = csv.DictWriter(f, fieldnames=columns)
        writter.writeheader()
        month = start_month
        year = start_year
        acc = 0
        while acc < duration:
            if month == 13:
                year += 1
                month = 1
            writter.writerow({'Year': year,
                              'Month': month,
                              'GMSL': output[acc]
                              })
            month += 1
            acc += 1


def calculate_end_date(start_year: int, start_month: int, duration: int) -> Tuple[int, int]:
    """Given the start year and start month and the duration in unit of month,
    return a tuple which first element is the end year and second is end month"""
    total_month = start_year * 12 + start_month + duration
    new_year = total_month // 12
    new_month = total_month % 12 + 1
    return (new_year, new_month)


def calculate_start_date(end_year: int, end_month: int, duration: int) -> Tuple[int, int]:
    """Given the end year and end month and the duration in unit of month,
    return a tuple which first element is the start year and second is start month"""
    total_month = end_year * 12 + end_month - duration
    new_year = total_month // 12
    new_month = total_month % 12 + 1
    return (new_year, new_month)

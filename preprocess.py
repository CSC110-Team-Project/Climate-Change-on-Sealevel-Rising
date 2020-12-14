"""CSC110 Final Project 2020: preprocess.py

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
import pandas as pd


class Preprocess:
    """The data processor of data sets.
    Functionalities:
    1. Processes the csv files to pandas data sets in a uniformed format
    2. Merge/combine the three data sets into one data set."""

    def sea_level_process(self, file: str) -> pd.DataFrame:
        """Process our sea level csv file to panda dataFrame that
        can be used by other modules.
        """
        dateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")
        sealvl = pd.read_csv(file, header=0, parse_dates=['Time'], date_parser=dateparser)
        sealvl['Year'] = sealvl['Time'].apply(lambda x: x.year)
        sealvl['Month'] = sealvl['Time'].apply(lambda x: x.month)
        sealvl = sealvl.drop(['Time'], axis=1)

        sealvl.head()
        return sealvl

    def seaice_process(self, file: str) -> pd.DataFrame:
        """Process our sea ice csv file to panda dataFrame that
        can be used by other modules.
        """
        seaice = pd.read_csv(file)
        seaice.columns = [x.strip() for x in list(seaice.columns)]

        grp = seaice.groupby(['Year', 'Month'])
        seaice = grp.agg("mean").reset_index()
        seaice.drop(columns=['Day', 'Missing'], inplace=True)

        seaice.head()
        return seaice

    def temperature_process(self, file: str) -> pd.DataFrame:
        """Process our sea ice csv file to panda DataFrame set that
        can be used by other modules.
        """
        dateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")
        temp = pd.read_csv(file, header=0, parse_dates=['dt'], date_parser=dateparser)
        temp['Year'] = temp['dt'].apply(lambda x: x.year)
        temp['Month'] = temp['dt'].apply(lambda x: x.month)
        temp = temp[['Year', 'Month', 'LandAverageTemperature']]

        temp.head()
        return temp

    def merge_data(self, seaIce: pd.DataFrame, temp: pd.DataFrame, seaLvl: pd.DataFrame) -> pd.DataFrame:
        """ Merge the three panda dataFrames. Return a dataset that has
        sea ice, temperature, sea level all in one.
        """
        data = seaLvl.merge(seaIce, how='inner').merge(temp, how='inner')
        data.head()
        return data

"""Module containing models representing catchment data.

The Model layer is responsible for the 'business logic' part of the software.

Catchment data is held in a Pandas dataframe (2D array) where each column contains
data for a single measurement site, and each row represents a single measurement
time across all sites.
"""

import pandas as pd
import numpy as np
from functools import reduce


def read_variable_from_csv(filename):
    """Reads a named variable from a CSV file, and returns a
    pandas dataframe containing that variable. The CSV file must contain
    a column of dates, a column of site ID's, and (one or more) columns
    of data - only one of which will be read.

    :param filename: Filename of CSV to load
    :return: 2D array of given variable. Index will be dates,
             Columns will be the individual sites
    """
    dataset = pd.read_csv(filename, usecols=['Date', 'Site', 'Rainfall (mm)'])

    dataset = dataset.rename({'Date': 'OldDate'}, axis='columns')
    dataset['Date'] = [pd.to_datetime(x, dayfirst=True) for x in dataset['OldDate']]
    dataset = dataset.drop('OldDate', axis='columns')

    newdataset = pd.DataFrame(index=dataset['Date'].unique())

    for site in dataset['Site'].unique():
        newdataset[site] = dataset[dataset['Site'] == site].set_index('Date')["Rainfall (mm)"]

    newdataset = newdataset.sort_index()

    return newdataset


def daily_total(data):
    """Calculate the daily total.

    :param data: 2D data array (Index must be np.datetime64 compatible format.)
    :returns: Daily total
    """
    return data.groupby(data.index.date).sum()


def daily_mean(data):
    """Calculate the daily mean.

    :param data: 2D data array (Index must be np.datetime64 compatible format.)
    :returns: Daily total
    """
    return data.groupby(data.index.date).mean()


def daily_max(data):
    """Calculate the daily max.

    :param data: 2D data array (Index must be np.datetime64 compatible format.)
    :returns: Daily total
    """
    return data.groupby(data.index.date).max()


def daily_min(data):
    """Calculate the daily min.

    :param data: 2D data array (Index must be np.datetime64 compatible format.)
    :returns: Daily total
    """
    return data.groupby(data.index.date).min()


def data_normalise(data):
    """Normalise any given 2D data array"""
    max_array = np.array(np.max(data, axis=0))
    return data / max_array[np.newaxis, :]


def daily_above_threshold(site_id, data, threshold):
    """Checks whether data is above a given threshold"""
    return_list = map(lambda x: (x > threshold), data[site_id])
    return return_list


def data_above_threshold(site_id, data, threshold):
    """"""

    def count_above_threshold(a, b):
        if b:
            return a + 1
        else:
            return a

    above_threshold = map(lambda x: x > threshold, data[site_id])
    return reduce(count_above_threshold, above_threshold, 0)


def sum_of_data(site_id, data):
    return reduce((lambda a, b: a + b), data[site_id])


class Location:
    def __init__(self, name):
        self.name = name


class Site(Location):
    version = "0.1"

    def __init__(self, name):
        super().__init__(name)
        self.measurements = {}

    def add_measurement(self, measurement_id, data):
        if measurement_id in self.measurements.keys():
            self.measurements[measurement_id] = \
                pd.concat([self.measurements[measurement_id], data])
        else:
            self.measurements[measurement_id] = data
            self.measurements[measurement_id].name = measurement_id

    @classmethod
    def get_version(cls):
        return "version "+str(cls.version)

    @staticmethod
    def create_sample_site():
        return Site('sample')

    def __str__(self):
        return self.name

    @property
    def last_measurements(self):
        return pd.concat(
            [self.measurements[key][-1:]
             for key in self.measurements.keys()], axis=1).sort_index()


class MeasurementSeries:
    def __init__(self, series, name, units):
        self.series = series
        self.name = name
        self.units = units
        self.series.name = self.name

    def add_measurement(self, data):
        self.series = pd.concat([self.series, data])
        self.series.name = self.name

    def __str_(self):
        if self.units:
            return f"{self.name} ({self.units})"
        else:
            return self.name


class Book:
    version = "0.1"

    def __init__(self, title, author):
        self.title = title
        self.author = author

    def __str__(self):
        return self.title + " by " + self.author

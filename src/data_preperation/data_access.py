from abc import ABC, abstractmethod
import pandas as pd

class DataWrapper(ABC):
    """Provides linearly interpolated data points and unified access by date to data of varying formats.

    Children should overwrite the init function."""

    @abstractmethod
    def __init__(self, path, noise=None):
        """Loads data"""
        self.data = None
        self.description = None

    def __getitem__(self, dt):
        """Provide access to data-table rows"""
        if dt in self.data.index:
            return self.data.loc[dt]
        else:
            dt_before = self.data.index[self.data.index.get_indexer([dt], method='ffill')[0]]
            dt_after = self.data.index[self.data.index.get_indexer([dt], method='backfill')[0]]
            if dt_after == dt_before:
                return self.data.loc[dt_before]
            w = (dt - dt_before) / (dt_after - dt_before)
            return (1-w) * self.data.loc[dt_before] + w * self.data.loc[dt_after]

    def __getattr__(self, attr):
        """Expose description rows (such as mean and std) as attributes."""
        if attr == 'loc':
            raise AttributeError('This is not a pandas dataframe, loc is not needed.')
        return self.description.loc[attr]


class WeatherData(DataWrapper):
    def __init__(self, path):
        self.data = pd.read_csv(path, sep=';', index_col='date', parse_dates=True)
        self.description = pd.read_csv(f'{path[:-4]}_description.csv', sep=';', index_col=0)

class EnergyData(DataWrapper):
    def __init__(self, path):
        self.data = pd.read_csv(path, sep=';', index_col=0, parse_dates=True)
        self.data.drop(['Date', 'Time of day'], axis=1, inplace=True)
        self.description = pd.read_csv(f'{path[:-4]}_description.csv', sep=';', index_col=0)

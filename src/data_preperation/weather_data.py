import tqdm
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from wetterdienst.provider.dwd.observation import DwdObservationRequest
from wetterdienst import Settings

import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='''
Fetch and format weather data. This requires downloading and saving tens of gigabytes of data!
''')

parser.add_argument('target_dir')
parser.add_argument('--load', dest='load', default=False, action='store_true')
parser.add_argument('--keep-split', dest='keep_split', default=False, action='store_true')
args = parser.parse_args()

def dt_to_np(dt: datetime):
    """Helper function to convert datetime to datetime64"""
    return np.datetime64(dt.replace(tzinfo=None))


def main():
    Path(args.target_dir).mkdir(parents=True, exist_ok=True)
    stations_path = os.path.join(args.target_dir, "stations_df.pkl")
    values_base = os.path.join(args.target_dir, "raw")
    split_base = os.path.join(args.target_dir, "split")

    columns_of_interest = ['humidity', 'radiation_global', 'radiation_sky_long_wave',
                           'radiation_sky_short_wave_diffuse', 'sun_zenith_angle', 'sunshine_duration',
                           'temperature_air_mean_200', 'wind_direction', 'wind_speed']
    dates = pd.to_datetime(['2015-01-01 00:00:00', '2021-12-31 23:59:59'])
    start_date, end_date = dates.tz_localize('Europe/Berlin').tz_convert('UTC').to_pydatetime()

    if not args.load:
        print('WARNING: Downloading and processing takes about 50GB of free disk space,',
              'a decent internet connection and a few hours of time.')

        # prepare request for DWD database
        Settings.tidy = True  # default, tidy data
        Settings.humanize = True  # default, humanized parameters
        Settings.si_units = True  # default, convert values to SI units
        request = DwdObservationRequest(
            parameter=["solar", "wind", "air_temperature"],
            resolution="10_minutes",
            start_date=start_date,  # if not given timezone defaulted to utc
            end_date=end_date,  # if not given timezone defaulted to utc
        )

        # stations that have data on our parameters
        stations_df = request.all().df
        stations_df.to_pickle(stations_path)
        print(stations_df)

        stations_ids = sorted(stations_df['station_id'])
        for id in stations_ids:
            try:
                print('trying station', id)
                values_df = request.filter_by_station_id(station_id=(id,)).values.all().df
            except:
                print('skipping', id)
                continue

            # only keep information we really need
            values_df = values_df.drop('quality', axis=1)
            values_df = values_df[values_df['parameter'].isin(columns_of_interest)].reset_index(drop=True)
            values_df['station_id'] = values_df['station_id'].astype('category')
            values_df['parameter'] = values_df['parameter'].astype('category')
            if values_df['value'].isnull().all():
                continue

            # save to disk for now
            values_df.to_pickle(os.path.join(values_base, f"values_df_{id}"))

    if not args.keep_split:
        print('splitting data into monthly bits')
        stations_df = pd.read_pickle(stations_path)

        value_paths = []
        for file in os.listdir(values_base):
            if not file.startswith('values_df_'):
                raise ValueError(f"That file shouldn't be there: {file}")
            path = os.path.join(values_base, file)
            value_paths.append(path)

        # split every data of every station into monthly bits
        all_values = None
        for path in tqdm.tqdm(sorted(value_paths)):
            id = path[-5:]  # file has name values_df_<id>
            values_df = pd.read_pickle(path)

            # streamline data. redundant if run from scratch
            values_df = values_df.drop('quality', axis=1)
            values_df['station_id'] = values_df['station_id'].astype('category')
            values_df['parameter'] = values_df['parameter'].astype('category')
            if values_df['value'].isnull().all():
                continue

            values_df['month'] = values_df['date'].apply(lambda x: x.strftime('%Y-%m'))
            for month, part in values_df.groupby('month'):
                month_path = os.path.join(split_base, month)
                Path(month_path).mkdir(parents=True, exist_ok=True)
                part = part.drop('month', axis=1).reset_index(drop=True)
                part.to_pickle(os.path.join(month_path, path[-5:]))

    if True:
        print('reassembling')
        all_mean = None
        all_std = None
        # reassemble bits of common month
        for month in tqdm.tqdm(sorted(os.listdir(split_base))):
            month_path = os.path.join(split_base, month)
            month_parts = None

            # concat information from all stations of the given month
            for station in sorted(os.listdir(month_path)):
                station_path = os.path.join(month_path, station)
                part = pd.read_pickle(station_path)
                month_parts = pd.concat([month_parts, part], ignore_index=True, sort=False)

            # then aggregate, so we only have mean and std for pairs (date, parameter)
            month_mean = month_parts.groupby(['date', 'parameter'])['value'].aggregate('mean').unstack()
            month_std = month_parts.groupby(['date', 'parameter'])['value'].aggregate('std').unstack()
            all_mean = pd.concat([all_mean, month_mean], sort=False)
            all_std = pd.concat([all_std, month_std], sort=False)

        all_mean.to_pickle(os.path.join(args.target_dir, "weather_mean_df.pkl"))
        all_std.to_pickle(os.path.join(args.target_dir, "weather_std_df.pkl"))

        # save final table to disk
        all_mean.to_csv(os.path.join(args.target_dir, "weather_mean.csv"), sep=';')
        all_mean.describe().to_csv(os.path.join(args.target_dir, "weather_mean_description.csv"), sep=';')
        all_std.to_csv(os.path.join(args.target_dir, "weather_std.csv"), sep=';')
        all_std.describe().to_csv(os.path.join(args.target_dir, "weather_std_description.csv"), sep=';')

        print(all_mean)
        print(all_std)
        print('These columns have NaNs in them:')
        print(all_mean.isna().any())


if __name__ == '__main__':
    main()

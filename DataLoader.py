"""
Gabo Bernardino, Niccolò Fabbri - 11/02/2024

Data loader for implied vol data
"""

import polars as pl
import os
import datetime as dt
import joblib
from copy import deepcopy
from typing import List, Optional, Union


class DataLoader(object):
    """
    Data loader to read in implied vol data for a list of dates.
    Can read from parquet or csv, depending on initialization
    
    Parameters
    ----------
    base_path: str
        Path to the data directory
    extension: str, default=None
        File extension of the datasets. Must be one of 'parquet' or 'csv'.
        If None, defaults to parquet
    """
    def __init__(self,
                 base_path: str,
                 extension: Optional[str] = None):
        self.base_path: str = base_path
        if extension is None or extension not in {'parquet', 'csv'}:
            self.extension = 'parquet'
        else:
            self.extension: str = extension
        self._reader_func = pl.read_csv if extension == 'csv' else pl.read_parquet
        
    def load_dates(self,
                   dates: List[Union[str, dt.date]],
                   n_jobs: int = 1) -> pl.DataFrame:
        """
        Read and merge dataframes for given dates
        
        Parameters
        ----------
        dates: list of str or datetime.date
            Dates for which to read the data. If str, must be in format 'YYYYmmdd'
        n_jobs: int, default=1
            Number of parallel workers to use (with joblib)
        """
        if not isinstance(dates[0], str):
            _dates = [x.strftime("%Y%m%d") for x in dates]
        else:
            _dates = deepcopy(dates)
            
        def _read_date(dd):
            return self._reader_func(f"{self.base_path}/impvol_{dd}.{self.extension}")
        
        if n_jobs == 1:
            return pl.concat([
                _read_date(dd) for dd in _dates
            ])
        else:
            out = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_read_date)(dd) for dd in _dates
            )
            return pl.concat(out)
        
        def dump(self, merge_df: pl.DataFrame, date_name: str, n_jobs: int = 1) -> None:
            _dates = merge_df[date_name].unique().sort().to_list()
            
            def _dump_day(dd):
                _dd = dd.strftime("%Y%m%d")
                pl.write_parquet(f"{self.base_path}/impvol_{_dd}.{self.extension}")
                return
            
            for dd in _dates:
                _dump_day(dd)

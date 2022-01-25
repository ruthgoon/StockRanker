# pylint: disable=import-error
import ast
import datetime
import json
import os
import random
import time
import zlib
from timeit import default_timer as timer

import modin.pandas as pd
import numpy as np
import redis

from .LoaderComponents import Components
from .RankDB import RankDB


class DataLoader:

    def __init__(self, save_directory=False):
        """
        The DataLoader is responsible for loading the Predictive GAN models and
        creating and keeping track of the ranked training data.

        Parameters:
            - save_directory :: The full path to the directory to save the ranked training data
        """
        self.save_path = "/path/to/save/path/" if not save_directory else save_directory
        self.components = Components()
        self.db = RankDB()
        self.redis = redis.Redis(host="localhost")

    def create_ranking(self, periods=False, num_workers=12, is_live=False):
        """
        Creates a series of weekly rankings and extracts, ranks, and serializes the features for every stock that exists
        for that week. Uses a multiprocess pool worker

        Parameters:
            - period (list) :: A list of lists with each sublist consisting of 2 datetimes, one the start of the week and one the
                               end of the week
        Returns:
            - True if the ranking has finished, False if it hasn't
        """

        if periods and not isinstance(periods, list):
            raise ValueError("Period must be of type list")

        if periods:
            # type check the period to make sure its not invalid or anything

            if len(periods) == 0:
                raise ValueError("Periods must have length >= 1")
            if not isinstance(periods[0], list):
                raise ValueError(
                    "Period must contain sublists containing 2 indices each")

        if not periods:

            # auto generate the periods going from todays date to 2015-01-01 (HERE)
            end_date = datetime.datetime.strptime("2021-01-01", "%Y-%m-%d")
            today = datetime.datetime.now()
            delta = abs((today - end_date).days)
            periods = []
            prev_date = False
            for i in range(0, delta, 7):
                prev_date = today - datetime.timedelta(days=7)
                periods.append([prev_date, today])
                today = prev_date

        periodic_data = []

        for p in periods:
            results = self.db.get_stock_dataframes(timeframe=p, live=is_live)
            periodic_data.append(results)

        for dataframes, period in zip(periodic_data, periods):
            # start the extraction for each frame in the dataframes list
            tickers = {}
            for dataframe in dataframes:
                comps = {}

                if len(dataframe) < 5:
                    continue

                comps.update(self.components.stock_component(dataframe))
                comps.update(self.components.technical_component(dataframe))
                # comps.update(self.components.market_component(dataframe)) fix once article_sentiment_cache is populated
                tickers[str(dataframe["ticker"][0])] = comps

            if not tickers:
                continue
            ranked_output = self.heuristic_rank(tickers)
            save_fp = self.save_path + \
                "{}_{}.npy".format(period[0].strftime(
                    "%Y-%m-%d"), period[1].strftime("%Y-%m-%d"))
            np.save(save_fp, ranked_output, allow_pickle=False)

    def _extraction_worker(self, dframe_object):
        """
        Takes a dataframe object and does some work on it
        TODO: multiprocessing put off until later because its fucky
        """
        num_secs = random.randint(5, 15)
        time.sleep(num_secs)
        return len(list(dframe_object.keys()))

    def heuristic_rank(self, feature_dictionary, weight_list=False):
        """
        Heuristic rank takes a weekly dict of tickers, ranks them based on the parameters defined
        below and then returns a sorted list of dicts

        Parameters:
            - dataset (dict) :: The dictionary of features

        Returns:
            -
        """
        weights = [[0.3, 0.3, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2], [0.5, 0.5]
                   ] if not weight_list else weight_list

        if not isinstance(weights, list):
            raise ValueError("`weight_list` must be of type list")

        resultants = []
        for ticker, val in feature_dictionary.items():

            # numpy conversion of vectors
            val = {x: np.array(val[x]) for x in val if x not in [
                "highest_price", "lowest_price", "avg_open", "avg_close"]}

            # (1) Stock Price Components
            # --------------------------

            # magnitudes between the mean and the max/min values
            max_mean_magnitude = (val["stock_prices"].max(
                axis=0) - np.mean(val["stock_prices"], axis=0)) / (val["stock_prices"].max(
                    axis=0))

            min_mean_magnitude = (np.mean(
                val["stock_prices"], axis=0) - val["stock_prices"].min(axis=0)) / np.mean(
                val["stock_prices"], axis=0)

            # mean growth
            mean_growth = np.mean(val["percent_growth"][1:], axis=0)

            # total stock score
            stock_score = (np.sum(max_mean_magnitude) * weights[0][0]) - (np.sum(min_mean_magnitude) * weights[0][1]) + (
                np.sum(mean_growth) * weights[0][2]) - (np.sum(val['avg_volatility']) * weights[0][3])

            # (2) Technical Components
            # --------------------------

            # calculate the magnitude of divergence between the MACD indicator and the signal
            div_mag = (val["macd"][1:] - val["signal"][1:]) / val["macd"][1:]

            # bollinger spread (wider == more volatile narrow == less volatile)
            boll_spread = np.linalg.norm(
                val["boll_upper"][1:] - val["boll_lower"][1:])

            # distances between boll bands & high/low prices

            high_boll_dist = np.linalg.norm(
                val["stock_prices"][:, [1]][1:] - np.column_stack(val["boll_upper"][1:]))

            low_boll_dist = np.linalg.norm(
                val["stock_prices"][:, [2]][1:] - np.column_stack(val["boll_lower"][1:]))

            # average the rsi-5, weigh it and then multiply by the abs of the r-value
            adjusted_rsi_factor = (100 - np.average(
                val["rsi-5"][:5], weights=[0.075, 0.075, 0.1, 0.2, 0.5])) * np.abs(val["rsi-5"][7])

            # TODO incorporate more technical indicators here too lazy to do the maff

            # calculate technical indicator component score
            tech_score = (np.sum(div_mag) * weights[1][0]) - (
                np.sum(boll_spread) * weights[1][1]) + ((high_boll_dist - low_boll_dist) * weights[1][2]) + (adjusted_rsi_factor * weights[1][3])

            combined_weighted = (
                stock_score * weights[2][0]) + (tech_score * weights[2][1])

            resultants.append(
                [ticker, stock_score, tech_score, combined_weighted])

        # sort the resultant list
        resultants.sort(key=lambda x: x[3])

        return resultants

    def unpack_data(self, timeframe=False, is_compressed=True, sum_predicts=False, normalize_vals=False):
        """
        DEPRECATED
        Unserializes all the .vtf data files and returns a single numpy array that represents each
        stock ticker

        Parameters:
            timeframe (list) :: a list of length 2 with index 0 being the starting time and index 1 being
            the ending time. Both datetime objects and strings in the format %Y-%m-%d are accepted

            is_compressed (bool) :: True if the data needs to be uncompressed before being read into memory

            sum_redicts (bool) :: If True, the predictive components (which are numpy arrays) will be converted
                                  to floats by finding their average volatility and normalizing. Needed when
                                  constructing the graph.

            normalize_vals (bool) :: If True, all values will be normalized

        Returns:
            ranking_list (list) :: A multidimensional list. The first index corresponding to each week index
                                   the second being a list of features, sorted from highest ranked to lowest

        """
        # check to see if the directory exists first
        if not self.save_path or not os.path.isdir(self.save_path):
            raise Exception("save_path required to unpack vectors")

        if timeframe:
            if type(timeframe) != list:
                raise Exception("timeframe must be a list")

            if len(timeframe) != 2:
                raise Exception("timeframe must be of length 2")
            if type(timeframe[0]) == str and type(timeframe[1]) == str:
                timeframe = [datetime.datetime.strptime(
                    d, "%Y-%m-%d") for d in timeframe]

        dir_files = [f for f in os.listdir(self.save_path) if os.path.isfile(
            os.path.join(self.save_path, f))]
        dir_files.sort()
        ranking_list = []

        # load the ticker shortcodes
        ticker_map = self._ticker_tokenizer(False)

        for f in dir_files:
            if ".vtf" not in f:
                continue
            date_lists = f.replace(".vtf", "").split("_")
            date_lists = [datetime.datetime.strptime(
                s, "%Y-%m-%d") for s in date_lists]

            if timeframe:
                if date_lists[0] < timeframe[0] or date_lists[1] > timeframe[1]:
                    continue

            full_path = os.path.join(self.save_path, f)
            # open the full path for reading
            with open(full_path, "rb") as f:
                raw = f.read()

            if is_compressed:
                raw = zlib.decompress(raw)
            raw = raw.decode('utf-8')
            raw = raw.replace('nan', 'np.nan')

            week_dict = eval(raw)
            weekly_ranking = {}
            # clean the weekly dict up and construct the numpy array
            for i, ranked in enumerate(week_dict['ranking']):
                ranking = [ranked['rank'], ranked['score']]
                for ci, component_key in enumerate(list(ranked['components'].keys())):
                    val = ranked['components'][component_key]
                    if not val:
                        continue
                    if isinstance(val, bytearray):
                        # deserialize bytearray
                        try:
                            val = self._deserialize_vector(val)
                            if sum_predicts:
                                val = np.sum(val)
                        except Exception as e:
                            continue

                    if isinstance(val, list):
                        val = np.array(val)
                        if sum_predicts:
                            val = np.sum(val)

                    ranking.append(val)

                # sort the ranking by the
                ranking = np.array(ranking)
                # convert the ticker to a ticker code
                if ranked['ticker'] not in list(ticker_map.keys()):
                    continue
                ticker_code = ticker_map[ranked['ticker']]
                weekly_ranking[ticker_code] = ranking
            ranking_list.append(weekly_ranking)
        return ranking_list

    def output_letor(self, output_path=False):
        """
        Takes a list of weeks, or the output of the unpacker, and transforms it into
        LETOR formatted files that it saves the file to the output_path
        """
        return False

    def _merge(self, master, d):
        """
        Merges dict d into dict master, returns dict master.
        """
        if not d:
            return master

        for k, v in d.items():
            master[k] = v

        return master

    def _feature_weight(self, feature_map):
        """
        Takes a feature_map (the dict of features for each given stock) and returns a weighted
        float score
        """
        score = 0.0

        items = len(list(feature_map.keys()))

        for key in list(feature_map.keys()):
            val = feature_map[key]

            if isinstance(val, pd.DataFrame):
                if val.empty:
                    items = items - 1
                    continue
                val = val.mean()
                val = float(val[0])

            elif type(val) == list:
                val = sum(val) / len(val)

            elif type(val) == float or type(val) == int:
                if key == 'avg_volatility':
                    val = -val

            else:
                items = items - 1
                continue

            score = score + val

        if len(list(feature_map.keys())) > 0:
            score = score / items
        return float(score)

    def _serialize_vector(self, arr: np.array) -> str:
        """
        Serializes a numpy array
        """
        arr_dtype = bytearray(str(arr.dtype), 'utf-8')
        arr_shape = bytearray(','.join([str(a) for a in arr.shape]), 'utf-8')
        sep = bytearray('|', 'utf-8')
        arr_bytes = arr.ravel().tobytes()
        to_return = arr_dtype + sep + arr_shape + sep + arr_bytes
        return to_return

    def _deserialize_vector(self, serialized_arr: str) -> np.array:
        """
        Deserializes an array from redis into the np.array type
        """
        sep = '|'.encode('utf-8')
        i_0 = serialized_arr.find(sep)
        i_1 = serialized_arr.find(sep, i_0 + 1)
        arr_dtype = serialized_arr[: i_0].decode('utf-8')
        arr_shape = tuple(
            [int(a) for a in serialized_arr[i_0 + 1:i_1].decode('utf-8').split(',')])
        arr_str = serialized_arr[i_1 + 1:]
        arr = np.frombuffer(arr_str, dtype=arr_dtype).reshape(arr_shape)
        return arr

    def _split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def _serialize_dict(self, d, compress=False):
        return json.dumps(d)

    def _deserialize_dict(self, serialized_dict, decompress=False):
        if decompress:
            serialized_dict = zlib.decompress(serialized_dict)

        return_dict = ast.literal_eval(serialized_dict)
        return return_dict

    def _ticker_tokenizer(self, ticker, auto_update=False):
        """
        Takes a alphanumeric ticker and returns a 4 integer long id unique to that ticker. Auto updates
        depending on whether or not any new tickers were added to the system. Returns the full ticker
        dictionary if ticker=False

        Parameters:
            - ticker (str | False) :: The ticker to generate the id for
            - auto_update (bool) :: If true, it will sync with the database and automatically update the
                                    ticker list. Super slow
        """
        ticker_mapping = os.path.join(self.save_path, "ticker_mapping")
        has_mapping = os.path.isfile(ticker_mapping)
        if auto_update or not has_mapping:
            self._make_ticker_mapping()

        mapping = {}
        with open(ticker_mapping, "r") as f:
            for line in f:
                line = line.replace("\n", "").split(" ")
                mapping[line[0].strip()] = line[1].strip()

        if not ticker:
            return mapping

        if ticker in list(mapping.keys()):
            return mapping[ticker]
        return False

    def _make_ticker_mapping(self):
        map_path = os.path.join(self.save_path, "ticker_mapping")
        stock_list = self.db.get_stock_list()
        if os.path.isfile(map_path):
            os.remove(map_path)
        i = 0
        with open(map_path, "w") as f:
            for ticker in stock_list:
                pad = f"{i:04}"
                f.write(f"{ticker} {pad}\n")
                i = i + 1
        return True

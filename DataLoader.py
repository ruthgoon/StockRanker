# pylint: disable=import-error
from .LoaderComponents import Components
from .RankDB import RankDB
import json
import os
import time
import redis
import faulthandler
import zlib
import datetime
import ast
import pandas as pd
import numpy as np


class AbstractTransformer(ast.NodeTransformer):
    """
    Because ast.literal_eval cannot parse the byte() primitive
    type in python, we need to institute this abstract syntax tree
    hack to brute force the AST to visit our bytes
    """
    ALLOWED_NAMES = set(['Bytes', 'None', 'False', 'True'])
    ALLOWED_NODE_TYPES = set([
        'Expression',
        'Tuple',
        'Call',
        'Name',
        'Load',
        'Constant',
        'Str',
        'Bytes',
        'Num',
        'List',
        'Dict',
        'UnaryOp',
        'USub'
    ])

    def visit_name(self, node):
        if not node.id in self.ALLOWED_NAMES:
            raise RuntimeError(
                "Name access to {} is not allowed".format(node.id))

        return self.generic_visit(node)

    def generic_visit(self, node):
        nodetype = type(node).__name__
        if nodetype not in self.ALLOWED_NODE_TYPES:
            raise RuntimeError("Invalid nodetype {}".format(nodetype))

        return ast.NodeTransformer.generic_visit(self, node)


class DataLoader:

    def __init__(self, ranking_save_path=False):
        """
        The DataLoader is responsible for loading the Predictive GAN models and 
        creating and keeping track of the ranked training data.

        Parameters:
            - model_save_path :: The full path to the directory to save the ranked training data 
        """
        self.save_path = ranking_save_path
        self.ast_transformer = AbstractTransformer()
        self.components = Components()
        self.db = RankDB()
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        with open("{}/DataLoaderConfig.json".format(self.cwd)) as f:
            self.config = json.load(f)

    def create_ranking(self, tickers=False, periods=False, db_cache=True, cache_location=False, newest_first=False):
        """
        This function handles the main ranking system by going through each available week (or starting
        at and ending at the specified period) and computing the following for each week:
            - Stock Price Component (High, Low, Average Open, Average Close, Growth W/W, Rate of Return LIS, Volatility)
            - Technical Component (MACD, Bolling Band, EMA, RSI, Stochastic Oscillator, Accumulative Distribution, Ease of Movement, CCI, Daily Log Return, Volume Returns)
            - Market News Component (Avg Sentiment, Causal Link Exploration, Movement Predictors)
            - Predictive Component (TSGAN, ARIMA, TreeModel, Stumpy Matrix, Discounted Cash Flow, Anomalous Activity)


        Each stock will be ranked according to each other stock and the stocks that heuristically have 
        grown the most, are the most stable, have the most stable and highest market opinion sentiment, and
        are predicted to grow and to be stable are ranked higher than those that do not fulfill these attributes.
        These metaparameters can be tweaked in the DataLoaderConfig.json configuration file.

        Once ranked, each stock can then be clustered together will like stocks to make portfolios/benchmarks 
        of like stocks. In total, there are 26 features that each stock possesses.

        TODO
            - finish tsgan and add it to the Predictive Components
            - fix ARIMA and add it to the Predictive Components
            - Finish CausalLink, MovementPredictor and at them to the Market News Component

        Parameters:
            - tickers :: A list of tickers to rank. If False, ALL tickers will be ranked

            - period :: A list with the 0th index being the starting date and the 1st index being the ending. If
                        False, the latest week for all tickers listed will be used

            - db_cache :: If true, each stocks rank for each week will be cached to the database. If False, they will
                       outputted to the cache_location

            - cache_location :: If db_cache is false this denotes the directory location where the ranked weekly
                                training data will be stored


        Returns:
            - Boolean - True if successfully executed, False if not
        """
        # faulthandler.enable()
        debug_logger = redis.Redis(host='localhost', port=6379, db=0)
        if type(tickers) == str:
            tickers = tickers.split(",")

        if not tickers:
            tickers = self.db.get_stock_list()

        # if a time period is not specified we start from the starting time period
        # defined in the config and go to the present. We then splice the dates so
        # we're left with an array of tuples with each tuple being a start and end date
        # (inclusive)
        if not periods:
            periods = []
            date_start = datetime.datetime.strptime(
                self.config['General']['OldestTickerTimestamp'], "%Y-%m-%d")
            date_end = datetime.datetime.now()
            num_days = (date_end - date_start).days
            last_date = date_start
            for i in range(0, num_days, 7):
                new_date = date_start + datetime.timedelta(days=i)
                week_list = [
                    last_date,
                    new_date
                ]
                last_date = new_date
                periods.append(week_list)

            if newest_first:
                periods = periods[::-1]

        else:
            if type(periods) == list:
                formatted_periods = []
                for period in periods:
                    if len(period) != 2:
                        raise Exception("Malformed datetime sublist")
                    start = datetime.datetime.strptime(period[0], "%Y-%m-%d")
                    end = datetime.datetime.strptime(period[1], "%Y-%m-%d")
                    formatted_periods.append([start, end])

                periods = formatted_periods
            else:
                raise Exception("periods must be a list")

        # indx 33
        for period in periods:

            # we need to pad 3 week intervals for the technical indicators to be working properly.
            # we only use the tech indicators corresponding with the week_start and week_end days

            week = {
                'week_start': datetime.datetime.strftime(period[0], "%Y-%m-%d"),
                'week_end': datetime.datetime.strftime(period[1], "%Y-%m-%d"),
                'tech_week_start': datetime.datetime.strftime(period[0] + datetime.timedelta(days=5), "%Y-%m-%d"),
                "tech_week_end": datetime.datetime.strftime(period[1] + datetime.timedelta(days=5), "%Y-%m-%d"),
                'num_stocks': 0,
                "ranking": []
            }

            comps = {}
            ranking = {}
            unsorted = []
            start_time = time.time()
            for ticker in tickers:
                debug_logger.set("dload", "{} - {}".format(period, ticker))
                # check to see if there are results for the stock

                has_results = self.db.has_prices(
                    ticker, week['week_start'], week['week_end'])

                if not has_results:
                    continue

                week['num_stocks'] = week['num_stocks'] + 1
                components = {}
                # calculate each component
                calculated_comps = {
                    "predictive": self.components.predictive_component(
                        ticker, week['week_start'], week['week_end']),
                    "market": self.components.market_component(
                        ticker, week['week_start'], week['week_end']),
                    "technical": self.components.technical_component(
                        ticker, week['tech_week_start'], week['tech_week_end']),
                    "stock": self.components.stock_component(
                        ticker, week['week_start'], week['week_end'])
                }

                # combine into 1 dict and get the weighted score and serialize
                for key in list(calculated_comps.keys()):
                    components = self._merge(
                        components, calculated_comps[key])

                weighted_score = self._feature_weight(components)

                ranking = {
                    "rank": -1,
                    "ticker": ticker,
                    "score": weighted_score,
                    "components": components
                }
                unsorted.append(ranking)

            # sort the unsorted list of tickers by their "score" keys
            unsorted = sorted(unsorted, key=lambda k: k['score'], reverse=True)
            sorted_vals = []

            # serialize
            for i, val in enumerate(unsorted):
                val['rank'] = i + 1
                for veckey in list(val['components'].keys()):

                    if isinstance(val['components'][veckey], pd.DataFrame):
                        val['components'][veckey] = self._serialize_vector(
                            val['components'][veckey].to_numpy())
                    if isinstance(val['components'][veckey], np.ndarray):
                        val = self._serialize_vector(val['components'][veckey])
                sorted_vals.append(val)

            # append it to the week object
            week['ranking'] = sorted_vals
            week['average_volume'] = -1.00

            # check to see if we need to cache the weekly record or if we write it to the output file
            if not db_cache:
                if not cache_location or not self.save_path:
                    raise Exception(
                        "cache_location must be specified if db_cache is False")

                spath = self.save_path if self.save_path else cache_location
                TEMP_DEBUG_FILE = "/home/dan/vestra/src/logs/dloader.log"
                save_filename = "{}_{}.vtf".format(
                    week['week_start'], week['week_end'])
                save_filename = spath + save_filename
                with open(save_filename, "wb") as f:
                    serialized = self._serialize_dict(week, compress=True)
                    f.write(serialized)

            else:
                self.db.cache_training_week(week)

    def unpack_data(self, timeframe=False, is_compressed=True, sum_predicts=False, normalize_vals=False):
        """
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
                            print(full_path)
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
        TODO FINISH THIS
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
        arr_dtype = serialized_arr[:i_0].decode('utf-8')
        arr_shape = tuple(
            [int(a) for a in serialized_arr[i_0 + 1:i_1].decode('utf-8').split(',')])
        arr_str = serialized_arr[i_1 + 1:]
        arr = np.frombuffer(arr_str, dtype=arr_dtype).reshape(arr_shape)
        return arr

    def _split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def _serialize_dict(self, d, compress=False):
        string = str(d)
        string_bin = string.encode()
        if compress:
            string_bin = zlib.compress(string_bin)
        return string_bin

    def _deserialize_dict(self, serialized_dict, decompress=False):
        if decompress:
            serialized_dict = zlib.decompress(serialized_dict)
        d = serialized_dict.decode('utf-8')
        d = d.replace('nan', 'np.nan')
        decoded_tree = ast.parse(d, mode='eval')
        self.ast_transformer.visit(decoded_tree)
        clause = compile(decoded_tree, '<AST>', 'eval')

        # this is the most dangerous bit of code in this entire class. if any bad data is
        # passed to this class during the deserialization of the ranked training data then
        # rce is possible
        decoded_dict = eval(clause, dict(Byte=bytes()))
        return decoded_dict

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

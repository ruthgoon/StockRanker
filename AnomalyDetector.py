# pyright: reportMissingImports=false, reportUnusedVariable=warning, reportUntypedBaseClass=error
import numpy as np
import datetime
import collections
from sklearn.ensemble import IsolationForest
from .RankDB import RankDB
from .StockEngine import StockEngine


class AnomalyDetector:

    def __init__(self, stock_list=False, results_to_return=750):
        """Detects anomalous stock activity given a stock list.

        Parameters:
             - stock_list (list) :: A list of tickers
             - results_to_return (int) :: How many resultant items to returnWWW

        """
        self.stocks = stock_list
        self.db = RankDB()
        self.stock_engine = StockEngine()
        self.VOLATILITY_FILTER = 0.05
        self.VOLUME_FILTER = 5e6
        self.NUM_TO_RETURN = results_to_return

    def run(self, time_period=False):
        """Runs the anomaly detector

        Returns:
             - results - A list of result dicts
        """
        features, historical_data, symbols, period = self._get_feature_map(
            time_period=time_period, volume_filter=self.VOLUME_FILTER)

        model = IsolationForest(n_estimators=100, random_state=0)
        model.fit(features)
        predictions = model.decision_function(features)
        results = [[predictions[i], symbols[i], historical_data[i]]
                   for i in range(0, len(predictions))]
        results = list(sorted(results))

        filtered_results = []
        for item in results[:self.NUM_TO_RETURN]:
            latest_date, todays_volume, average_vol_five_days, average_vol_twenty_days = self.stock_engine.volume_changes(
                item[2])
            volatility_vol_five_days, volatility_vol_twenty_days, volatility_all = self.stock_engine.volatility(
                item[2], dataframe=True)
            if average_vol_five_days == None or volatility_vol_five_days == None:
                continue
            filtered_results.append({
                'date': latest_date,
                'period': period,
                'ticker': item[1],
                'anomaly_score': item[0],
                'todays_volume': todays_volume,
                "avg_volume_5d": average_vol_five_days,
                'avg_volume_20d': average_vol_twenty_days,
                '5bar_volatility': volatility_vol_five_days,
                "20bar_volatility": volatility_vol_twenty_days
            })
        return filtered_results

    def _get_feature_map(self, time_period=False, volume_filter=False, volatility_filter=False):
        """
        Returns the feature map, historical price info, future prices and the symbol names for the securities
        time_period (list) :: A list with the first index being the start date and the second index being the end date
        """
        if not self.stocks:
            return False

        # Load the data from the stock dictionary
        features = []
        symbol_names = []
        historical_price_info = []

        if not time_period:
            today = datetime.datetime.now()
            # default to 2 financial weeks
            previous = today - datetime.timedelta(days=10)
            time_period = [previous, today]

        for stock in self.stocks:
            price_data = self.db.get_stock_dataframes([stock], time_period)

            if type(price_data) == bool and not price_data:
                continue
            if len(price_data) < 5:
                continue

            volatility_5, volatility_20, volatility_all = self.stock_engine.volatility(
                price_data, dataframe=True)

            if volatility_filter:
                if volatility_5[0] < volatility_filter:
                    continue

            stock_feature_dict = self.stock_engine.get_technical_indicators(
                price_data)

            if not stock_feature_dict:
                continue

            feature_list = []
            for key in list(sorted(stock_feature_dict.keys())):
                feature_list.extend(stock_feature_dict[key])

            if np.isnan(feature_list).any() == True:

                continue

            avg_volume = np.mean(list(price_data['volume'])[-30:])

            if volume_filter and avg_volume < volume_filter:
                continue

            features.append(feature_list)
            symbol_names.append(stock)
            historical_price_info.append(price_data)
            features, historical, symbols = self._preproc_data(
                features, historical_price_info, symbol_names)

        return features, historical, symbols, time_period

    def _preproc_data(self, features, historical_data, stock_names):
        length_dictionary = collections.Counter(
            [len(feature) for feature in features])
        length_dictionary = list(length_dictionary.keys())
        most_common_length = length_dictionary[0]

        filtered_features, filtered_historical_price, filtered_symbols = [], [], []
        for i in range(0, len(features)):
            if len(features[i]) == most_common_length:
                filtered_features.append(features[i])
                filtered_symbols.append(stock_names[i])
                filtered_historical_price.append(historical_data[i])

        return filtered_features, filtered_historical_price, filtered_symbols

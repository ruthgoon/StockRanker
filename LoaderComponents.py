# pylint: disable=import-error
import datetime
import math

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import (FFT, AutoARIMA, ExponentialSmoothing, Prophet,
                          RNNModel, Theta)

from .RankDB import RankDB
from .StockEngine import StockEngine


class Components:
    def __init__(self):
        """
        This class abstracts away the actual computation from the DataLoader class and provides a high-level
        API to generating all 4 of the DataLoaders neccessary components
        """
        self.db = RankDB()
        self.engine = StockEngine()

    def stock_component(self, ticker, week_start, week_end):
        """
        Gets weeks highest and lowest prices, average open and close, percent growth
        compared to the last weeks end, average volatility, and the longest increasing
        subsequence in stock price.

        Parameters:
            - ticker :: A string ticker
            - week_start :: The datetime object to start the week
            - week_end :: The datetime object ending the week

        Returns:
            - stock_dict :: A dictionary containing the following keys:
                            [highest_price, lowest_price, avg_open, avg_close, percent_growth,
                             avg_volatility, rate_of_return, lis]

        """

        # load the data
        stock_data = self.db.get_stock_prices(
            ticker, time_period=[week_start, week_end])

        if not stock_data:
            return False

        high = -1
        low = 9e50
        sums = [0.0, 0.0]   # indexes are: avg_open, avg_close
        closes = []
        vol_return = 0.0
        for stock in stock_data:
            if stock['high'] > high:
                high = stock['high']

            if stock['low'] < low:
                low = stock['low']

            sums[0] = sums[0] + stock['open']
            sums[1] = sums[1] + stock['close']
            closes.append(stock['close'])

        if stock_data[0]['close'] == 0:
            growth = 0
        else:
            growth = float((stock_data[len(stock_data) - 1]['close'] -
                            stock_data[0]['close']) / stock_data[0]['close'])

        avgs = [x / len(stock_data) for x in sums]
        std, vol = self.engine.volatility(closes)
        ror = self.engine.rate_of_return(closes)
        ror = 0 if len(ror) == 0 else (sum(ror) / len(ror))*100
        lis = self.engine._lis(closes)

        # construct the dictionary
        stock_dict = {
            "highest_price": high,
            "lowest_price": low,
            "avg_open": avgs[0],
            "avg_close": avgs[1],
            "percent_growth": growth,
            "avg_volatility": vol,
            "rate_of_return": ror,
            "lis": len(lis)}

        return stock_dict

    def technical_component(self, ticker, week_start, week_end):
        """
        Gets the MACD, Bolling bands and WMA technica indicators. Returns a stock dict with the 
        following keys:
            [macd, bolling, ema, rsi-5, stochastic-5, accum_dist, eom-5, cci-5, daily_log_return, volume_returns]
        """

        stock_data = self.db.get_stock_prices(ticker, [week_start, week_end])
        if not stock_data:
            return False

        close = [x['close'] for x in stock_data]
        macd, macd_signal, macd_labels = self.engine.macd_indicator(close)

        # this calculation gives us the average macd value over the week
        macd = sum([macd[x] for x in macd.keys()]) / len(macd.keys())

        # we will do the same for the RSI, Bolling and EMA values

        ema = self.engine.ema_indicator(close)
        ema = sum([ema[x] for x in ema.keys()]) / len(ema.keys())

        bol = self.engine.bol_indicator(close)
        bol = sum([bol[x] for x in bol.keys()]) / len(bol.keys())

        base_dict = {
            "macd": macd,
            "bolling": bol,
            "ema": ema
        }
        extended_dict = self.engine.get_technical_indicators(stock_data)
        if extended_dict and base_dict:
            tech_dict = {**extended_dict, **base_dict}
        elif base_dict:
            tech_dict = base_dict
        elif extended_dict:
            tech_dict = extended_dict
        else:
            tech_dict = {}

        return tech_dict

    def market_component(self, ticker, week_start, week_end):
        """
        The market component gets the average market sentiment, total number of 
        causal relations and the strength of those causal relations as well as 
        any prediction of stock movements. Returns a dictionary with the following
        keys:[avg_sentiment, num_causal_relations, causal_strength, movement_strength] 
        NOTE: Only avg_sentiment works for now! i still need to implement the causality
        models

        """
        sentiments = self.db.get_sentiment_cache(
            ticker, [week_start, week_end])

        if not sentiments or (type(sentiments) == list and len(sentiments) == 0):
            return False

        avg_sentiment = 0.0
        for sentiment in sentiments:
            avg_sentiment = avg_sentiment + float(sentiment['sentiment'])
        avg_sentiment = avg_sentiment / len(sentiments)
        market_dict = {
            "avg_sentiment": avg_sentiment,
            "num_causal_relations": None,
            "causal_strength": None,
            "movement_strength": None
        }
        return market_dict

    def predictive_component(self, ticker, week_start, week_end, num_days=20):
        """
        This component deals with the generating various predictions for the given stock if 
        they exist. This returns an aggregated average for each prediction method. It returns
        a dict with the following keys: [tsgan, prophet, fft, theta, rnn, exponential]. If the
        particular model is not available for any given model, None type is returned. 
        TODO: Add support for predicting all columns (not just the close column)

        Parameters:
            - ticker :: A string ticker
            - week_start :: A string of the start date (format YYYY-mm-dd HH:MM:SS)
            - week_end :: A string of the ending date
            - num_days :: How many days to extend the forecast to (default 20)


        """
        week_start = datetime.datetime.strptime(
            week_start, "%Y-%m-%d") - datetime.timedelta(20)
        week_end = datetime.datetime.strptime(
            week_end, "%Y-%m-%d") + datetime.timedelta(20)

        sdata = self.db.get_stock_prices(
            ticker, time_period=[week_start, week_end], time_normalized=True)

        if not sdata:
            return False

        window = sdata[int(len(sdata)*0.80)]['date']

        frame = pd.DataFrame(
            sdata, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        frame['date'] = pd.to_datetime(frame['date'])
        frame = frame.reset_index()

        # this is counterintuitive but the forecasting models don't accept NaN values so we
        # have to fill them with the mean values as to not throw the model off
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.fillna(frame.mean())
        try:
            series = TimeSeries.from_dataframe(frame, 'date', ['close'], 'B')
        except:
            return {
                "exponential": None,
                "prophet": None,
                "fft": None,
                "theta": None,
                "tsgan": None,
                "arima": None
            }
        #train, val = series.split_after(window)

        # do the predictions here (TODO: implement autoARIMA and the TSGAN better)
        forecast = {
            "exponential": self._exp_smoothing(series, num_days),
            "prophet": self._prophet(series, num_days),
            "fft": self._fft(series, num_days),
            "theta": self._theta(series, num_days),
            "tsgan": None,
            "arima": None
        }
        # transform the prediction to a dataframe
        for model in list(forecast.keys()):
            if not forecast[model]:
                continue
            df = forecast[model].pd_dataframe()
            forecast[model] = df
        return forecast

    def _auto_arima(self, train_series, input_series):
        """
        This module computes the automatic ARIMA values

        Parameters:
            - train_series :: A data set to train the ARIMA model on 
            - input_series :: The input series to use to extend the ARIMA prediction

        Returns:
            - prediction :: A timeseries that the AutoARIMA model predicted
        """
        try:
            model = AutoARIMA()
            model.fit(train_series)
            n = len(input_series)
            prediction = model.predict(n)
            return prediction

        except Exception as e:
            print("EXCEPTION_ARIMA_ERR_DEBUG_TRUE")
            return None

    def _exp_smoothing(self, train_series, num_days):
        try:
            model = ExponentialSmoothing()
            model.fit(train_series)
            prediction = model.predict(num_days)
            return prediction
        except Exception as e:
            print("EXCEPTION_EXP_ERR_DEBUG_TRUE")  # lazy, remove in prod
            return None

    def _prophet(self, train_series, num_days):
        try:
            prophet = Prophet()
            prophet.fit(train_series)
            prediction = prophet.predict(num_days)
            return prediction

        except Exception as e:
            print("EXCEPTION_PROPH_ERR_DEBUG_TRUE")
            return None

    def _fft(self, train_series, num_days):
        try:
            fft = FFT()
            fft.fit(train_series)
            prediction = fft.predict(num_days)
            return prediction

        except Exception as e:
            print('EXCEPTION_FFT_ERR_DEBUG_TRUE')
            return None

    def _theta(self, train_series, num_days):
        try:
            theta = Theta()
            theta.fit(train_series)
            prediction = theta.predict(num_days)
            return prediction
        except Exception as e:
            print(e)
            return None

# pylint: disable=import-error
import collections
import math
import random

import numpy as np
import pandas
import requests
import ta
import datetime
import yfinance as yf
from scipy.stats import linregress, norm
from stockstats import StockDataFrame

# pylint: disable=no-member


class StockEngine():
    '''
    The Stock Engine collects stock data, calculates financial indicators and
    performs stock screening analysis.
    '''

    def __init__(self):
        self.FUNDKEYS = {
            "name": "longName",
            "sector": "sector",
            "summary": "longBusinessSummary",
            "industry": "industry",
            "previous_close": "previousClose",
            "two_hundred_day_avg": "twoHundredDayAverage",
            "fifty_day_avg": "fiftyDayAverage",
            "market_cap": "marketCap",
            "total_assets": "totalAssets",
            "beta": "beta",
            "logo": "logo_url"
        }
        self.FINKEYS = {
            "research_development": "Research Development",
            "income_before_tax": "Income Before Tax",
            "net_income": "Net Income",
            "gross_profit": "Gross Profit",
            "ebit": "Ebit",
            "operating_income": "Operating Income",
            "interest_expense": "Interest Expense",
            "income_tax_expense": "Income Tax Expense",
            "total_revenue": "Total Revenue",
            "total_operating_expenses": "Total Operating Expenses",
            "cost_of_revenue": "Cost Of Revenue"
        }

        self.IEX_CONFIG = {
            "iex_test": "your-iex-test-token",
            "iex_prod": "your-iex-production-token",
            "selection": "cloud"
        }

    def get_fundamentals(self, ticker):
        """
        Returns the fundamentals for a given stock in a dict
        """
        try:
            fund = yf.Ticker(ticker)
            finfo = fund.info
            fund_keys = list(finfo.keys())
            return_obj = {}
            return_obj['ticker'] = ticker
            for key in list(self.FUNDKEYS.keys()):
                if self.FUNDKEYS[key] in fund_keys:
                    return_obj[key] = finfo[self.FUNDKEYS[key]]
            return return_obj
        except Exception as e:
            print("Exception occured with {}".format(ticker))
            return False

    def get_financials(self, ticker):
        """
        Fetches financial information pertaining to a ticker
        TODO: replace with iex API equivalent (yahoo api sucks)
        """
        try:

            stock = yf.Ticker(ticker)
            financials = stock.financials
            financials = financials.transpose()
            finlist = []
            for i, row in financials.iterrows():
                finobj = {}
                finobj['date'] = i
                finobj['ticker'] = ticker
                for key in list(self.FINKEYS.keys()):
                    index = self.FINKEYS[key]
                    if index in row:
                        finobj[key] = row[index]
                    else:
                        finobj[key] = False
                finlist.append(finobj)
            return finlist

        except Exception as e:
            return False

    def get_technical_indicators(self, df, n_timeframe=5):
        """
        Reads a dataframe and returns the technical indicators. These differ from the standalone
        TA functions (macd, rsi, etc.) in that they are optimized for dataframes
        """
        techs = {}

        # this is really inefficient and slow but due to my spaghetti code in the DataLoader this conversion
        # is necessary.
        if type(df) == list:
            df = pandas.DataFrame(df)

        # RSI & Stochastic indicators have the same history so we calculate them together
        hists = [5, 10, 15]
        for history in hists:

            rsi = ta.momentum.RSIIndicator(
                df['close'], window=history, fillna=True).rsi().values.tolist()
            stochastic = ta.momentum.StochasticOscillator(
                df['high'], df['low'], df['close'], window=history, smooth_window=int(history/3), fillna=True).stoch().values.tolist()

            rsi_slope, rsi_r_val, rsi_p_val = self.regressive_slope(
                rsi[-n_timeframe:])

            stochastic_slope, stoch_r_val, stoch_p_val = self.regressive_slope(
                stochastic[-n_timeframe:])

            techs["rsi-{}".format(history)] = rsi[-n_timeframe:] + \
                [rsi_slope, rsi_r_val, rsi_p_val]

            techs["stochastic-{}".format(history)] = stochastic[-n_timeframe:] + [
                stochastic_slope, stoch_r_val, stoch_p_val]

        # Accumulation Distribution
        dist = ta.volume.acc_dist_index(
            df['high'], df['low'], df['close'], df['volume'], fillna=True).values.tolist()
        dist = dist[-n_timeframe:]
        dist_slope, dist_r_val, dist_p_val = self.regressive_slope(dist)

        techs['accum_dist'] = [dist_slope, dist_r_val, dist_p_val]

        # Ease of Movement & CCI
        hists = [5, 10, 20]
        for history in hists:
            eom = ta.volume.ease_of_movement(
                df['high'], df['low'], df['volume'], window=history, fillna=True).values.tolist()
            cci = ta.trend.cci(df['high'], df['low'], df['close'],
                               window=history, constant=0.015, fillna=True).values.tolist()

            eom_slope, eom_r_val, eom_p_val = self.regressive_slope(
                eom[-n_timeframe:])
            cci_slope, cci_r_val, cci_p_val = self.regressive_slope(
                cci[-n_timeframe:])

            techs['eom-{}'.format(history)] = [eom_slope, eom_r_val, eom_p_val]
            techs['cci-{}'.format(history)] = cci[-n_timeframe:] + \
                [cci_slope, cci_r_val, cci_p_val]

        # Daily Log Return
        dr = ta.others.daily_return(
            df['close'], fillna=True).values.tolist()
        dlr = ta.others.daily_log_return(
            df['close'], fillna=True).values.tolist()
        techs['daily_return'] = dr[-n_timeframe:]
        techs['daily_log_return'] = dlr[-n_timeframe:]

        # volume difference
        volume_list = df['volume'].values.tolist()
        volume_list = [vol for vol in volume_list if vol != 0]
        volume_returns = [volume_list[x] / volume_list[x - 1]
                          for x in range(1, len(volume_list))]

        slope_vol, r_value_vol, p_value_vol = self.regressive_slope(
            volume_returns[-n_timeframe:])

        techs["volume_returns"] = volume_returns[-n_timeframe:] + [
            slope_vol, r_value_vol, p_value_vol]

        return techs

    def regressive_slope(self, df):
        x_axis = np.arange(len(df))
        if x_axis.size == 0:
            return 0, 0, 0
        regression_model = linregress(x_axis, df)
        slope, r_val, p_val = round(regression_model.slope, 3), round(
            abs(regression_model.rvalue), 3), round(regression_model.pvalue, 4)
        return slope, r_val, p_val

    def get_data(self, ticker, json=True, period="max"):
        """
        Gets data about a stock given the period and its ticker
        Returns a list of dicts
        """
        data = yf.Ticker(ticker)
        if json:
            return data.history(period=period).transpose().to_json()
        try:
            history = data.history(period=period)
            robject = []
            for i, row in history.iterrows():
                iter_obj = {}
                iter_obj['date'] = i
                iter_obj['open'] = float(row['Open'])
                iter_obj['high'] = float(row['High'])
                iter_obj['low'] = float(row['Low'])
                iter_obj['close'] = float(row['Close'])
                iter_obj['volume'] = float(row['Volume'])
                robject.append(iter_obj)
            return robject
        except:
            return False

    def iex_get_data(self, tickers, start_time=False, end_time=False):
        """
        Retrieves time series data from IEXCloud given the starting date
        and the ending date to get data from

        Parameters:
            - ticker (str|list)
            - start_time (datetime) :: The starting datetime 
            - end_time (date) :: The ending datetime
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        if not isinstance(tickers, list):
            raise Exception("tickers must be of type list")

        results = {}

        for tick in tickers:
            # generate the url & send the request
            endpoint = self._iex_api_url(tick, start_time, end_time)
            resp = requests.get(endpoint)
            json = resp.json()
            if not isinstance(json, list):
                # error parsing
                if "error" in json:
                    results[tick] = None
                continue

            results[tick] = []

            for r in json:
                results[tick].append({
                    "date": datetime.datetime.fromtimestamp(r['date']/1000.0),
                    "open": r['open'],
                    "high": r['high'],
                    "low": r['low'],
                    "close": r['close'],
                    "volume": r['volume']
                })
        return results

    def macd_indicator(self, timeseries):
        """
        Using the MACD signal line crossover indicator, this function
        first calculates the signal and MACD indicators and then returns
        whether to BUY, SELL or HOLD based on the timeseries. If the MACD
        crosses over the signal line (upward slope) it's indicative to BUY,
        if it dips below the signal line it's indicative to SELL and if it
        stays steady then HOLD is set for the timeframe.

        Returns: macd, signal, values

        NOTE: Timeseries must be a dataframe
        """
        if type(timeseries) == list:
            timeseries = pandas.DataFrame(timeseries, columns=['Close'])
        val = StockDataFrame.retype(timeseries)
        signal = val['macds']
        macd = val['macd']
        values = []
        for i in range(1, len(signal)):
            if macd[i] > signal[i] and macd[i - 1] <= signal[i - 1]:
                values.append(1)
            elif macd[i] < signal[i] and macd[i - 1] >= signal[i - 1]:
                values.append(-1)
            else:
                values.append(0)

        return macd.tolist(), signal.tolist(), values

    def rate_of_return(self, timeseries):
        """
        Calculates the rate of return over the timeseries.
        """
        ror = []
        if(isinstance(timeseries, list)):
            for i in range(1, len(timeseries)):
                if(timeseries[i-1] == 0):
                    ror.append(0)
                else:
                    ror.append((timeseries[i] / timeseries[i-1]) - 1)
            return ror

    def volume_changes(self, historical_price):

        def _fmt_num(value):
            if value < 1000:
                value = str(value)
            elif value >= 1000 and value < 1000000:
                value = round(value / 1000, 2)
                value = str(value) + "K"
            else:
                value = round(value / 1000000, 1)
                value = str(value) + "M"

            return value

        volume = list(historical_price["volume"])
        dates = list(historical_price["date"])
        dates = [str(date) for date in dates]

        # Get volume by date
        volume_by_date_dictionary = collections.defaultdict(list)
        for j in range(0, len(volume)):
            date = dates[j].split(" ")[0]
            volume_by_date_dictionary[date].append(volume[j])

        for key in volume_by_date_dictionary:
            volume_by_date_dictionary[key] = np.sum(
                volume_by_date_dictionary[key])

        all_dates = list(
            reversed(sorted(volume_by_date_dictionary.keys())))
        latest_date = all_dates[0]
        latest_data_point = list(reversed(sorted(dates)))[0]

        today_volume = volume_by_date_dictionary[latest_date]
        average_vol_last_five_days = np.mean(
            [volume_by_date_dictionary[date] for date in all_dates[1:6]])
        average_vol_last_twenty_days = np.mean(
            [volume_by_date_dictionary[date] for date in all_dates[1:20]])

        return latest_data_point, today_volume, average_vol_last_five_days, average_vol_last_twenty_days

    def volatility(self, timeseries, timeframe=252, dataframe=False):
        """
        Calculates the volatility of a timeseries. This function
        returns both the standard deviation as well as the normalized
        volatility (using 252 trading days as the normalization rate).

        Returns: std, norm_volatility
        """
        if dataframe:
            close_price = list(timeseries["close"])
            volatility_five_bars = np.std(close_price[-5:])
            if len(close_price) >= 20:
                volatility_twenty_bars = np.std(close_price[-20:])
            else:
                volatility_twenty_bars = False
            volatility_all = np.std(close_price)

            return volatility_five_bars, volatility_twenty_bars, volatility_all

        ror = self.rate_of_return(timeseries)
        std = np.std(ror)
        norm_volatility = math.sqrt(timeframe)*std
        return std.item(), norm_volatility.item()

    def value_at_risk(self, timeseries, iterations=5000, r_free_rate=0.0):
        """
        Calculates the value at risk of a timeseries by utilizing the
        black-scholes method to calculate an expected return of the
        timeseries and running a monte-carlo simulation. The number of
        iterations is defined by the iterations variable. This function
        assumes that the number of periods is the len(timeseries) (therefore)
        each indicie in the list should be the close value of that day

        Returns: percent_loss_95, percent_loss_99, simulated_returns
        """
        n_periods = len(timeseries)
        std, norm_volatility = self.volatility(timeseries)
        simulated_returns = []
        for i in range(iterations):
            inv_rand = random.random()
            b_scholes = math.exp((r_free_rate - 0.5*std**2)*(n_periods/252) +
                                 std*math.sqrt(n_periods/252)*norm.ppf(inv_rand))
            simulated_returns.append(b_scholes)

        percent_loss_95 = np.percentile(simulated_returns, 1 - .95)
        percent_loss_99 = np.percentile(simulated_returns, 1 - .99)
        return percent_loss_95.item(), percent_loss_99.item(), simulated_returns

    def top_ranked(self, db, threshold=50, num_returned=10):
        """
        Returns the top ranked stock databases. Top ranked stocks are ones
        with the longest increasing subsequence in stock price over the
        last `threshold` days, highest average and most recent stock value increase,
        lowest average volatility over the last `threshold` days and the one with
        the most recent BUY rating and a most recent UNDER rating.
        The categories are weighted as such:
            Stock price increase - 45%
            Low average volatility - 30%
            Most recent ratings - 25%

        Returns: A dictionary with each value being a stock ticker and its value being
        the rank of its performance.

        * TODO : Implement this with the LSTM timeseries forecasting function once complete from the
                 PredictionEngine class
        """
        ranked_output = {}
        formatted_output = {}

        stock_list = db.get_stock_list(page=-1)
        # iterate through our stock data, calculating the MACD/RSI rating, stock increase
        # and the volatility
        for stock in stock_list:
            stock_data = db.get_stock_data(stock)
            dates = [x['date'] for x in stock_data[stock]][::-1]
            dates = dates[:threshold]
            full_timeseries = [x
                               for x in stock_data[stock] if x['date'] in dates]
            full_timeseries = pandas.DataFrame(full_timeseries)
            timeseries = [x['close']
                          for x in stock_data[stock] if x['date'] in dates]
            volume = stock_data[stock][len(stock_data[stock]) - 1]['volume']

            # volume tweaker
            if volume < 5e5:
                continue

            lis = self._lis(timeseries)
            norm_lis = sum(lis) / volume

            current_volume = stock_data[stock][0]['volume']
            increase, rank = self._increase(timeseries)
            vol, adjusted_vol = self.volatility(timeseries)
            rsi, ratings = self.rsi_indicator(full_timeseries)

            rating = sum([1 for x in ratings if x == "UNDER"])
            rsi = list(rsi.values())[1:]
            rsi = [x for x in rsi if x == x]
            rank = (increase * 0.42) - (vol * 0.30) - ((1e5 - volume)*0.03) + (sum(lis)/len(lis))  # nopep8
            rank = rank - (sum(rsi)/len(rsi) * 0.25)

            ranked_output[stock] = rank
        return (sorted(ranked_output.items(), key=lambda x: x[1], reverse=True))

    def top_increase(self, db, period=5):
        """
        Calculates and returns stocks with the largest price increase given 
        the period. The price increase is then returned along with the ticker
        of the stock
        """

        stocks = db.get_stock_list(page=-1)
        master_stock_data = db.get_stock_data(stocks)
        top_increases = {}
        for stock in stocks:
            stock_data = master_stock_data[stock][::-1]
            stock_data = [x['close'] for x in stock_data][0:period]
            increase = -(stock_data[len(stock_data) - 1] -
                         stock_data[0]) / stock_data[0]
            top_increases[stock] = increase * 100
        return (sorted(top_increases.items(), key=lambda x: x[1], reverse=True))

    def rsi_indicator(self, timeseries):
        """
        Uses the relative strength index osillicator to gauge whether or
        not a timeseries is overvalued, undervalued or neither. An RSI
        value for a point 70 or above means the security is overbought or
        overvalued while an indicator of 30 or below means its oversold or
        undervalued. OVER, UNDER, HOLD.

        Returns: rsi, values
        """
        if type(timeseries) == list:
            timeseries = pandas.DataFrame(timeseries, columns=['Close'])
        val = StockDataFrame.retype(timeseries)
        rsi = val['rsi_6']
        values = []
        for i in range(1, len(rsi)):
            if rsi[i] >= 70:
                values.append("OVER")
            elif rsi[i] <= 30:
                values.append("UNDER")
            else:
                values.append("HOLD")
        return rsi.to_dict(), values

    def ema_indicator(self, timeseries):
        """
        Returns: ema
        """
        if type(timeseries) == list:
            timeseries = pandas.DataFrame(timeseries, columns=['Close'])
        val = StockDataFrame.retype(timeseries)
        ema = val['tema']
        return ema.tolist()

    def bol_indicator(self, timeseries):
        """
        Gets the bollinger bands upper, lower and median bounds

        Returns:
        upper, lower, boll (lists) :: The upper bound, lower bound and median
        """
        if type(timeseries) == list:
            timeseries = pandas.DataFrame(timeseries, columns=['Close'])

        val = StockDataFrame.retype(timeseries)
        return val["boll_ub"].tolist(), val["boll_lb"].tolist(), val["boll"].tolist()

    def df_lis(self, df, is_live=False):
        """
        Dataframe version of the lis
        """
        cols = ["open", "high", "low", "close", "volume"]
        results = []
        for col in cols:
            df_list = df[col].tolist()
            lis = self._lis(df_list)
            results.append(lis)
        return results

    def _lis(self, arr):
        """
        This function returns the longest increasing subssequence of an array.
        For example, given the array [30, 10, 20, 50, 40, 80, 60] the function
        will return [10, 20, 40, 60]. Finding the LIS is useful in timeseries 
        analysis for finding the overall trend of a timeseries. If the length of
        the LIS for one timeseries is higher than the other it indicates that the
        first timeseries has a higher growth-rate.
        Returns: LIS list
        """
        if not arr:
            return arr

        M = [None] * len(arr)
        P = [None] * len(arr)
        L = 1
        M[0] = 0

        for i in range(1, len(arr)):
            lower_bound = 0
            upper_bound = L
            if arr[M[upper_bound - 1]] < arr[i]:
                j = upper_bound
            else:
                while upper_bound - lower_bound > 1:
                    mid = (upper_bound + lower_bound) // 2
                    if arr[M[mid-1]] < arr[i]:
                        lower_bound = mid
                    else:
                        upper_bound = mid
                j = lower_bound
            P[i] = M[j-1]
            if j == L or arr[i] < arr[M[j]]:
                M[j] = i
                L = max(L, j+1)
        result = []
        position = M[L-1]
        for _ in range(L):
            result.append(arr[position])
            position = P[position]

        return result[::-1]

    def _increase(self, array):
        """
        Calculates the largest percent increase between elements in the 
        array and returns the increase along with the rank of the increase.
        If the increase is between the first and second items in the array 
        the rank is 0, if its between 1-2 it's 1, etc.
        Returns: increase_rate, rank
        """
        increase_rate = -1
        rank = -1
        for i, value in enumerate(array):
            if i == 0 or array[i-1] == 0:
                continue
            pchange = (value - array[i-1]) / array[i-1]
            if pchange > increase_rate:
                increase_rate = pchange
                rank = i
        return increase_rate, rank

    def _iex_api_url(self, ticker, start_date=False, end_date=False):
        base_url = "https://{}.iexapis.com/stable/time-series/HISTORICAL_PRICES/{}"
        base_url = base_url.format(self.IEX_CONFIG['selection'], ticker)

        if start_date:
            if isinstance(start_date, datetime.datetime):
                start_date = datetime.datetime.strftime(start_date, "%Y-%m-%d")
            base_url += "?from={}".format(start_date)

        if end_date:
            if isinstance(end_date, datetime.datetime):
                end_date = datetime.datetime.strftime(end_date, "%Y-%m-%d")
            base_url += "?" if not start_date else "&"
            base_url += "to={}".format(end_date)

        base_url += "?" if not (start_date or end_date) else "&"
        base_url += "token={}".format(self.IEX_CONFIG['iex_prod'])

        return base_url

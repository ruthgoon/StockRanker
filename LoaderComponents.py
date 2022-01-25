# pylint: disable=import-error
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

    def stock_component(self, stock_dataframe, is_live=False):
        """
        Gets weeks highest and lowest prices, average open and close, percent growth
        compared to the last weeks end, average volatility, and the longest increasing
        subsequence in stock price.

        Parameters:
            - stock_dataframe (Dataframe) :: The dataframe of the stock for that week

        Returns:
            - stock_dict :: A dictionary containing the following keys:
                            [stock_prices, highest_price, lowest_price, avg_open, avg_close, percent_growth,
                             avg_volatility, rate_of_return, lis]


            "stock_prices": stock_data,
            "highest_price": high,
            "lowest_price": low,
            "avg_open": avgs[0],
            "avg_close": avgs[1],
            "percent_growth": growth,
            "avg_volatility": vol,
            "lis": len(lis)

        """
        stock_dict = {
            "stock_prices": stock_dataframe[[
                "open", "high", "low", "close", "volume"]].to_numpy().tolist(),
            "highest_price": stock_dataframe["high"].max(),
            "lowest_price": stock_dataframe["low"].min(),
            "avg_open": stock_dataframe["open"].mean(),
            "avg_close": stock_dataframe["close"].mean(),
            "percent_growth": stock_dataframe[["open", "high", "low", "close", "volume"]].pct_change().fillna(value=0).to_numpy().tolist(),
            "avg_volatility": stock_dataframe["close"].rolling(window=2).std().fillna(value=0).to_numpy().tolist(),
            "lis": self.engine.df_lis(stock_dataframe)
        }
        return stock_dict

    def technical_component(self, stock_dataframe):
        """
        Takes a stock dataframe and returns the following technical indicators:
            - macd 
            - signal
            - macd_ratings
            - boll
            - boll_upper
            - boll_lower
            - ema
            - rsi
            - stochastic
            - accum_dist
            - eom
            - cci

        Parameters:
            - stock_dataframe (Dataframe) :: The dataframe of stock data

        Returns:
            - tech_dict (dict) :: A dictionary with each computed technical component value
        """
        macd, signal, macd_ratings = self.engine.macd_indicator(
            stock_dataframe)

        boll_u, boll_l, boll = self.engine.bol_indicator(stock_dataframe)
        tech_dict = {
            "macd": macd,
            "signal": signal,
            "macd_ratings": macd_ratings,
            "boll": boll,
            "boll_upper": boll_u,
            "boll_lower": boll_l,
            "ema": self.engine.ema_indicator(stock_dataframe)
        }
        tech_dict.update(self.engine.get_technical_indicators(stock_dataframe))

        return tech_dict

    def market_component(self, dataframe, period):
        """
        Extracted market sentiment, news confidence and causality index (coming soon)

        Parameters:
            - dataframe (Dataframe) :: The stock dataframe
            - period (list) :: The period to get. The first index of the list is the start date and the 
                               second is the end date
        Returns:
            - sentiment (list) :: A list of the aggregated daily sentiment for the stock (if it exists for the period) False
                                  if it doesnt exist
        """
        ticker = dataframe['ticker'][0]

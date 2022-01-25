import pymysql
import datetime
from pandas import DataFrame
import numpy as np
from dbutils.steady_db import connect


class RankDB():
    """
    a mariadb wrapper for the following tables:
        - stock_data
        - corporate_data
        - corporate_financials
        - model_event_queue
        - top_performers
    """

    def __init__(self):

        self.credentials = ['localhost', 'username',
                            'password', 'db']
        self.db = connect(
            creator=pymysql,
            host=self.credentials[0],
            user=self.credentials[1],
            password=self.credentials[2],
            database=self.credentials[3],
            autocommit=True,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

    def get_stock_dataframes(self, tickers=False, timeframe=False, limit=False, live=False):
        """
        Returns a dataframe of stock OHLCV information.

        Parameters:
            - tickers (False|list) :: If False, every ticker available will be used
            - timeframe (False|list) :: If defined, should be a list with the first value being the start date and the 
                                        second vaue being the end date. The timevalues should be either strings or datetime
                                        values. If live=false, they should be in the form %Y-%m-%d. If live=True, they
                                        should be in the format %Y-%m-%d_%H:%M:%S
            - limit (False|int) :: How many results to return. If False all results will be returned
            - live (Boolean) :: If true, data from the live_stock_data table will be queried. If False, data from the 
                                stock_data_table is queried

        Returns:
            - frames (dict) :: A dictionary of Dataframes with each key being the ticker of the corresponding dataframe.
                               Dataframes have the following columns: [date, open, high, low, close, volume]
        """
        cursor = self.db.cursor()

        if not tickers:
            tickers = self.get_stock_list()
            if not tickers:
                raise ValueError("Tickers not found, DB Issue??")

        if live:
            select_query = "SELECT `ticker`, `timestamp`, `open`, `bid`, `bid_size`, `ask`, `ask_size`, `close`, `volume`, `market_cap`, `change`, `percent_change` FROM `live_stock_data` WHERE `ticker` = %s"
        else:
            select_query = "SELECT `ticker`, `date`, `open`, `high`, `low`, `close`, `volume` FROM `stock_data` WHERE `ticker` = %s "

        if timeframe:
            # TODO
            # this does not deal with live stock data rewrite the last 2 placeholders to allow for granular time
            # increments/decrements of 5 seconds
            select_query += 'AND (`{}` BETWEEN "{}" AND "{}")'.format(
                "date" if not live else "timestamp", timeframe[0].strftime("%Y-%m-%d"), timeframe[1].strftime("%Y-%m-%d"))

        limit_string = ";" if not limit else " LIMIT {};".format(limit)

        select_query += " ORDER BY `{}` DESC{}".format(
            "date" if not live else "timestamp", limit_string)

        frames = []
        for ticker in tickers:

            cursor.execute(select_query, ticker)
            results = cursor.fetchall()

            if not results:
                continue

            # convert each result to a dataframe
            frames.append(DataFrame(results).set_index(
                "date" if not live else "timestamp"))
        return frames

    def get_stock_list(self):
        """
        Returns every stock as a list
        """
        cursor = self.db.cursor()
        sql = "SELECT `ticker` FROM `corporate_info`;"
        cursor.execute(sql)
        results = cursor.fetchall()
        cursor.close()
        results = [x['ticker'] for x in results]
        return results

    def has_prices(self, ticker, week_start, week_end):
        """
        Checks to see if there exists a price for a stock at the given time period.
        returns true if there are results and false is there arent
        """
        sql = "SELECT `id` FROM `stock_data` WHERE `ticker` = '{}' AND `date` BETWEEN '{}' AND '{}' LIMIT 1;".format(
            ticker, week_start, week_end)

        cursor = self.db.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        return len(results) != 0

    def cache_training_week(self, ranking_object):

        rank_keys = ['week_start', 'week_end',
                     'average_volume', 'num_stocks', 'ranking']
        if set(ranking_object.keys()) != set(rank_keys):
            return False

        r = ranking_object
        cursor = self.db.cursor()
        insertion_query = "INSERT INTO `weekly_ranking_data`(`id`, `week_start`, `week_end`, `average_volume`, `num_stocks`, `ranking`)"
        insertion_query += " VALUES (NULL, %s, %s, %s, %s, %s);"

        cursor.execute(insertion_query, (r['week_start'], r['week_end'],
                                         r['average_volume'], r['num_stocks'], r['ranking']))
        return True

    def get_training_weeks(self):

        # the total number of training week
        """
        Returns a tuple of start and end dates
        """
        sql = "SELECT `week_start`, `week_end` FROM `weekly_ranking_data` ORDER BY `week_start`;"
        cursor = self.db.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        training_periods = [[result['week_start'], result['week_end']]
                            for result in results]
        return training_periods

    def get_sentiment_cache(self, ticker, time_period=False):
        """
        Retrieves the sentiments given an articles publish date and a ticker value

        Parameters:
            - ticker :: The ticker of which to retrieve the sentiment information from 
            - time_period :: A list with the 0th index being the start date and the 1st
                             index corresponding to the end date. if not specified all 
                             sentiments will be returned

        Returns:
            - sentiment_cache :: A list of dicts from the start date to the end date (if specified)
                                 with the following keys: [published_on, sentiment]

        """
        cursor = self.db.cursor()

        query = "SELECT `published_on`, `sentiment` from `article_sentiment_cache` WHERE `ticker` = %s"

        if time_period:
            start = time_period[0]
            end = time_period[1]
            query = query + \
                " AND `published_on` BETWEEN %s AND %s".format(start, end)

        query = query + " ORDER BY `published_on` DESC;"

        if time_period:
            cursor.execute(query, (ticker, start, end))
        else:
            cursor.execute(query, ticker)
        results = cursor.fetchall()
        return results

    def test(self):
        cursor = self.db.cursor()
        d1 = datetime.datetime.strptime("2021-07-24", "%Y-%m-%d")
        d2 = datetime.datetime.strptime("2021-07-25", "%Y-%m-%d")

        q1 = "SELECT COUNT(*) FROM article_sentiment_cache WHERE published_on BETWEEN '2021-07-24' AND '2021-07-25';"
        q2 = "SELECT COUNT(*) FROM article_sentiment_cache WHERE published_on BETWEEN %s AND %s;"

        cursor.execute(q1)
        results = cursor.fetchall()
        cursor.execute(q2, (d1, d2))
        res2 = cursor.fetchall()

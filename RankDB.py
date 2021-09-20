import pandas as pd
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
                            'password', 'vestra']
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

    def get_stock_prices(self, ticker, time_period=False, limit=False, time_normalized=False, dataframe=False):
        """
        Gets the stock prices given a ticker and the time period
        TODO: Add filters

        Parameters:
            - ticker :: The ticker for which to get the stock data (string)
            - time_period :: a list with the 0th index being the starting time string and the
                             first index being the ending date string (not inclusive). Note*
                             that the strings must be in the "YYYY-MM-DD HH:mm:ss" format
            - time_normalized :: If True, all gaps in the data will be normalized with NaN values
            - limit :: The limit is how many values to
            - dataframe :: If true, a pandas dataframe object will be returned
        """
        cursor = self.db.cursor()
        sql = "SELECT `date`, `open`, `high`, `low`, `close`, `volume` FROM `stock_data` WHERE `ticker` = '{}'".format(
            ticker)

        if time_period:

            if type(time_period) != list or len(time_period) != 2:
                return False

            if not time_period[0]:

                earliest_timestamp = "SELECT `date` FROM `stock_data` WHERE `ticker` = '{}' ORDER BY `date` ASC LIMIT 1;".format(
                    ticker)
                cursor.execute(earliest_timestamp)
                results = cursor.fetchall()

                if not results:
                    raise Exception(
                        "Ticker value `{}` not found".format(ticker))

                time_period[0] = results[0]['date']
                time_period[0] = time_period[0].strftime(
                    "%Y:%m:%d 00:00:00")

            sql = sql + \
                " AND `date` BETWEEN '{}' AND '{}'".format(
                    time_period[0], time_period[1])
        if limit:
            sql = sql + " LIMIT {}".format(limit)
        sql = sql + " ORDER BY `date` DESC;"
        cursor.execute(sql)
        results = cursor.fetchall()
        cursor.close()

        if not results:

            return False

        if time_normalized:

            result_datetimes = [x['date'] for x in results]
            norm_results = []
            times = [results[0]['date'], results[len(results) - 1]['date']]

            diff = (times[0] - times[1]).days
            for d in range(diff):
                dt = times[len(times) - 1] + datetime.timedelta(days=d)
                if dt not in result_datetimes:
                    norm_results.append({
                        "ticker": ticker,
                        "date": dt,
                        "open": np.nan,
                        "high": np.nan,
                        "low": np.nan,
                        "close": np.nan,
                        "volume": np.nan})
                else:
                    idx = result_datetimes.index(dt)
                    norm_results.append(results[idx])
            results = norm_results

        if dataframe:
            results = DataFrame(results)

        return results

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
            print(list(ranking_object.keys()))
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

        query = "SELECT `published_on`, `sentiment` from `article_sentiment_cache` WHERE match(ticker) against (%s in boolean mode)"

        if time_period:
            start = time_period[0]
            end = time_period[1]
            query = query + \
                " AND `published_on` BETWEEN %s AND %s".format(start, end)

        query = query + " ORDER BY `published_on` DESC;"

        if time_period:
            cursor.execute(query, (ticker, start, end))

            results = cursor.fetchall()

        else:
            cursor.execute(query, ticker)
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
        print(results[0]['COUNT(*)'])
        print(res2[0]['COUNT(*)'])

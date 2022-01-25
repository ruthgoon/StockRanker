from numpy.lib.arraysetops import isin
from numpy.lib.index_tricks import nd_grid
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneebow.rotor import Rotor
import numpy as np
import datetime


class StockCluster():
    def __init__(self, cluster_input=False):
        """
        The ClusterModel class is a DBSCAN model that creates a graph of each stock for each week and clusters
        stocks based on their feature vectors, grouping like stocks together.

        Parameters:
            - cluster_input (list|False) :: If a list is passed it will autoload the clusterer
        """
        self.input_data = cluster_input
        self.labels = False

    def cluster(self, input_val, labels, week_start, week_end, eps=False, min_samples=False):
        """
        Uses DBSCAN to create the cluster

        Parameters:
            - eps (bool|float) :: The Epsilon value for DBSCAN. If False it will be set automatically
            - min_samples (bool|float) :: The number of samples to produce. If False, it will be set automatically

        Returns:

            - clusters (list[dict]) :: Returns a cluster object for each week given the shape of 
                                       the input data. A cluster object is a dictionary with the 
                                       following keys: 
                                        [
                                            week_start (str),
                                            week_end (str),
                                            num_cluster (int),
                                            num_noise (int),
                                            stocks (list)
                                        ]

                                        The "stocks" key is a list of stock dicts with the following
                                        keys:
                                        [
                                            token (str), 
                                            data (numpy.ndarray),
                                            cluster (int)
                                        ]

                                        Where the token is the ticker token for that stock, data is the 
                                        data vector for that given data point and the cluster is the ID
                                        of the cluster to which the stock belongs to (-1 for noise). 

        """
        if not isinstance(input_val, np.ndarray):
            raise Exception("Input data not set")

        if not min_samples:
            min_samples = 4

        if eps and type(eps) == float or type(eps) == int:
            eps = [eps for _ in range(len(input_val))]

        if type(week_start) != str or type(week_end) != str:
            raise Exception("Week start/week end must be str")

        # generate the dbscan clusters for each week in the input data
        clusters = []
        current_week = week_start

        for i, week in enumerate(input_val):
            weekly_labels = labels[i]
            week = week[:, ~np.isnan(week).any(axis=0)]
            # week should be an np array of shape (n_stocks, n_features)
            if not eps:
                epsilon = self.optimize_eps(week)
            else:
                epsilon = eps[i]

            # dbscan
            model = DBSCAN(eps=epsilon, min_samples=min_samples).fit(week)

            num_clusters = len(set(model.labels_)) - \
                (1 if -1 in model.labels_ else 0)

            weekly_end_date = self._add_days_to_string(current_week)
            weekly_cluster = {
                "week_start": current_week,
                "week_end": weekly_end_date,
                "eps": epsilon,
                "num_samples": min_samples,
                "num_clusters": num_clusters,
                "num_noise": sum([1 if i == -1 else 0 for i in model.labels_]),
                "stocks": []
            }
            current_week = weekly_end_date

            for i, label in enumerate(model.labels_):
                weekly_vector = week[i]
                weekly_cluster['stocks'].append({
                    "token": weekly_labels[i],
                    "data": weekly_vector,
                    "cluster": label
                })

            clusters.append(weekly_cluster)

        return clusters

    def optimize_eps(self, input_data=False):
        """
        This function attempts to optimize the epsilon value for DBSCAN. Epsilon is the distance that one point
        must be to another for them to be considered neighbours. Optimizing epsilon involves finding the distance
        of any 2 neighbours, ordering in ascending order of distance, and finding the point of maximum curvature.

        Returns:
            - optimized_eps (list[float]) :: A list of optimized epsilon value
        """

        if type(input_data) == bool:
            if not input_data:
                input_data = self.input_data
            else:
                raise Exception("Cannot accept true as input_data")

        elif type(input_data) != np.ndarray:
            raise Exception("Input data must be numpy array")

        try:
            rotor = Rotor()
            rotor.fit_rotate(input_data)
            idx = rotor.get_elbow_index()
            return input_data[idx]

        except Exception as e:
            print(e)
            return 50

    def optimize_num_samples(self, total_num_entities):
        return int(0.0133333333 * int(total_num_entities))

    def _format_samples(self, norm_vals=False):
        """
        Formats the self.input dictionary to be a numpy array of size (n_weeks, n_stocks, n_features),
        returns it along with a list of strings containing the ticker codes. Also sets self.input_data 
        and self.labels to be the output when run automatically

        Returns:
            - labels, features (np.array, np.array) :: the label and feature np arrays
        """
        if not self.input_data:
            raise Exception("Input required to be loaded")
        weekly_vectors = []
        weekly_labels = []
        for week in self.input_data:
            week_keys = list(week.keys())
            ticker_labels = []
            num_features = len(week[week_keys[0]])
            vect = np.zeros((len(week_keys), 33))
            for i, ticker in enumerate(week_keys):
                components = week[ticker]
                for k, comp in enumerate(components):
                    try:
                        vect[i][k] = comp
                    except:
                        continue
                ticker_labels.append(ticker)
            weekly_vectors.append(vect)
            weekly_labels.append(ticker_labels)

        if norm_vals:
            unorm = weekly_vectors
            weekly_vectors = []
            for week in unorm:
                normed = week / np.linalg.norm(week)
                weekly_vectors.append(normed)
        week_vector = np.array(weekly_vectors)
        self.input_data = week_vector
        self.labels = weekly_labels
        return week_vector, weekly_labels

    def _check_date_type(self, date, date_format):
        """
        Checks to see if a string obliges by a datetime format. Type safe as well

        Parameters:
            - date (str) :: A string representation of a datetime object
            - date_format (str) :: The format to check against
        """

        if type(date) != str:
            date = str(date)

        try:
            test_date_object = datetime.datetime.strptime(date, date_format)
            return test_date_object
        except:
            return False

    def _add_days_to_string(self, input_string, n_days=7):
        """
        Takes a date string formatted in the form "%Y-%m-%d" and adds n_days to it.
        Returns a date string in the same form
        """
        date_obj = self._check_date_type(input_string, "%Y-%m-%d")
        if not date_obj:
            raise Exception("Date string improperly formatted")
            # the error above should literally never happen

        date_obj = date_obj + datetime.timedelta(days=n_days)
        return datetime.datetime.strftime(date_obj, "%Y-%m-%d")

    def set_input(self, input_data):
        if type(input_data) != list:
            raise Exception("Input data must be in list form")
        self.input_data = input_data

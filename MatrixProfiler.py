
import stumpy
from numba import cuda


class MatrixProfiler:

    def __init__(self):
        self.gpu_device = [x.id for x in cuda.list_devices()]

    def cuda_compute_matrix(self, timeseries, window_size=10):
        """
        Takes a list of dicts and returns a numpy matrix using cuda

        Parameters:
            - timeseries (np.array) :: A numpy array with the shape (5,n) or (7,n)
            - window_size (int) :: The size of the window

        Returns:
            - matrix_profile (np.array) :: The multidimensional matrix
            - matrix_indicies (np.array) :: The index values of the multidimensional array
        """

        matrix_profile, matrix_indicies = stumpy.gpu_stump(
            timeseries, m=window_size, device_id=self.gpu_device)

        return matrix_profile, matrix_indicies

    def cpu_compute_multidim_matrix(self, timeseries, window_size=10, discords=True):
        """
        Same as above but timeseries can be a multidimesional timeseries object.
        uses the cpu tho no cuda support

        Discords:
            - If true, anomalies and their indexes will be returned rather than motifs
        """

        matrix_profile, matrix_indicies = stumpy.mstump(
            timeseries, m=window_size, discords=discords)

        return matrix_profile, matrix_indicies

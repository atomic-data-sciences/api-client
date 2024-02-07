from uuid import UUID

from monty.json import MSONable
from pandas import DataFrame

from atomicds.results import RHEEDImageResult


class RHEEDVideoResult(MSONable):
    def __init__(
        self,
        data_id: UUID | str,
        timeseries_data: DataFrame,
        cluster_image_data: list[RHEEDImageResult] | None,
        snapshot_image_data: list[RHEEDImageResult] | None,
    ):
        """RHEED video result

        Args:
            data_id (UUID | str): Data ID for the entry in the data catalogue.
            timeseries_data (DataFrame): Pandas DataFrame with timeseries data associated with the video.
                Includes cluster assignments, specular intensity, strain, etc...
            cluster_image_data (list[RHEEDImageResult]): List of RHEEDImageResult objects containing data for
                images associated with each identified cluster in the video.
            snapshot_image_data (list[RHEEDImageResult]): List of RHEEDImageResult objects containing data for
                images associated with each user extracted snapshot in the video.
        """
        self.data_id = data_id
        self.timeseries_data = timeseries_data
        self.cluster_image_data = cluster_image_data
        self.snapshot_image_data = snapshot_image_data

    def get_plot(self):
        # TODO: Implement
        pass

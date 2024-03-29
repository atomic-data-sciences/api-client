from __future__ import annotations

import os
from datetime import datetime
from io import BytesIO
from typing import Literal

import networkx as nx
from pandas import DataFrame
from PIL import Image
from tqdm.auto import tqdm

from atomicds.core import BaseClient
from atomicds.results import RHEEDImageResult, RHEEDVideoResult, XPSResult


class Client(BaseClient):
    """Atomic Data Sciences API client"""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = "https://api.atomicdatasciences.com/",
        mute_bars: bool = False,
    ):
        """
        Args:
            api_key (str | None): API key. Defaults to None which will try and pull from the ADS_API_KEY environment variable.
            endpoint (str): Root API endpoint. Will prioritize pulling from the ADS_API_ENDPOINT environment variable.
                If none provided it defaults to 'https://api.atomicdatasciences.com/'.
            mute_bars (bool): Whether to mute progress bars. Defaults to False.
        """
        api_key = api_key or os.environ.get("ADS_API_KEY")
        endpoint = os.environ.get("ADS_API_ENDPOINT") or endpoint

        if api_key is None:
            raise ValueError("No valid ADS API key supplied")

        self.mute_bars = mute_bars

        super().__init__(api_key=api_key, endpoint=endpoint)

    def search(
        self,
        keywords: str | list[str] | None = None,
        include_organization_data: bool = True,
        data_ids: str | list[str] | None = None,
        data_type: Literal[
            "rheed_image", "rheed_stationary", "rheed_rotating", "xps", "all"
        ] = "all",
        status: Literal["success", "pending", "error", "running", "all"] = "success",
        growth_length: tuple[int | None, int | None] = (None, None),
        upload_datetime: tuple[datetime | None, datetime | None] = (None, None),
        last_accessed_datetime: tuple[datetime | None, datetime | None] = (None, None),
    ) -> DataFrame:
        """Search and obtain data catalogue entries

        Args:
            keywords (str | list[str] | None): Keyword or list of keywords to search all data catalogue fields with.
                This searching is applied after all other explicit filters. Defaults to None.
            include_organization_data (bool): Whether to include catalogue entries from other users in
                your organization. Defaults to True.
            data_ids (str | list[str] | None): Data ID or list of data IDs. Defaults to None.
            data_type (Literal["rheed_image", "rheed_stationary", "rheed_rotating", "xps", "all"]): Type of data. Defaults to "all".
            status (Literal["success", "pending", "error", "running", "all"]): Analyzed status of the data. Defaults to "all".
            growth_length (tuple[int | None, int | None]): Minimum and maximum values of the growth length in seconds.
                Defaults to (None, None) which will include all non-video data.
            upload_datetime (tuple[datetime | None, datetime | None]): Minimum and maximum values of the upload datetime.
                Defaults to (None, None).
            last_accessed_datetime (tuple[datetime | None, datetime | None]): Minimum and maximum values of the last accessed datetime.
                Defaults to (None, None).
        Returns:
            (DataFrame): Pandas DataFrame containing matched entries in the data catalogue.

        """
        params = {
            "keywords": keywords,
            "include_organization_data": include_organization_data,
            "data_ids": data_ids,
            "data_type": None if data_type == "all" else data_type,
            "status": status,
            "growth_length_min": growth_length[0],
            "growth_length_max": growth_length[1],
            "upload_datetime_min": upload_datetime[0],
            "upload_datetime_max": upload_datetime[1],
            "last_accessed_datetime_min": last_accessed_datetime[0],
            "last_accessed_datetime_max": last_accessed_datetime[1],
        }

        data = self._get(
            sub_url="data_entries/",
            params=params,
        )
        column_mapping = {
            "uuid": "Data ID",
            "upload_datetime": "Upload Datetime",
            "last_accessed_datetime": "Last Accessed Datetime",
            "char_source_type": "Type",
            "raw_name": "File Name",
            "pipeline_status": "Status",
            "raw_file_type": "File Type",
            "source_name": "Instrument Source",
            "sample_name": "Sample Name",
            "growth_length": "Growth Length",
            "tags": "Tags",
            "name": "Owner",
        }
        catalogue = DataFrame(data)
        if len(catalogue):
            catalogue = catalogue.drop(columns=["user_id"])
        return catalogue.rename(columns=column_mapping)

    def get(
        self, data_ids: str | list[str]
    ) -> list[RHEEDVideoResult | RHEEDImageResult | XPSResult]:
        """Get analyzed data results

        Args:
            data_ids (str | list[str]): Data ID or list of data IDs from the data catalogue to obtain analyzed results for.

        Returns:
            (list[RHEEDVideoResult | RHEEDVideoResult | XPSResult]): List of result objects

        """
        if isinstance(data_ids, str):
            data_ids = [data_ids]

        data: list[dict] = self._get(  # type: ignore  # noqa: PGH003
            sub_url="data_entries/",
            params={
                "data_ids": data_ids,
                "include_organization_data": True,
            },
        )

        kwargs_list = []
        for entry in data:
            data_id = entry["uuid"]
            data_type = entry["char_source_type"]
            kwargs_list.append({"data_id": data_id, "data_type": data_type})

        # sort by submission order; this is important to match external labels
        kwargs_list = sorted(kwargs_list, key=lambda x: data_ids.index(x["data_id"]))

        pbar = (
            tqdm(
                desc="Obtaining data results",
                total=len(data),
            )
            if not self.mute_bars
            else None
        )

        return self._multi_thread(self._get_result_data, kwargs_list, pbar)

    def _get_result_data(
        self,
        data_id: str,
        data_type: Literal["xps", "rheed_image", "rheed_stationary", "rheed_rotating"],
    ) -> RHEEDVideoResult | RHEEDImageResult | XPSResult | None:
        if data_type == "xps":
            result: dict = self._get(sub_url=f"xps/{data_id}")  # type: ignore  # noqa: PGH003

            return XPSResult(
                data_id=data_id,
                xps_id=result["xps_id"],
                binding_energies=result["binding_energies"],
                intensities=result["intensities"],
                predicted_composition=result["predicted_composition"],
                detected_peaks=result["detected_peaks"],
                elements_manually_set=bool(result["set_elements"]),
            )

        if data_type == "rheed_image":
            return self._get_rheed_image_result(data_id)

        if data_type in ["rheed_stationary", "rheed_rotating"]:
            # Get timeseries data
            timeseries_data = self._get_rheed_timeseries_result(data_id)

            # Get cluster and extracted image data
            cluster_frames: list[dict] = self._get(  # type: ignore  # noqa: PGH003
                sub_url="data_entries/video_cluster_frames/",
                params={"data_ids": [data_id]},
            )

            extracted_frames: list[dict] = self._get(  # type: ignore  # noqa: PGH003
                sub_url=f"data_entries/video_single_frames/{data_id}",
            )

            def __obtain_frame_data(frames, metadata_fields):
                image_params = []

                for frame in frames.get("frames", []):
                    metadata = {
                        key: value
                        for key, value in frame.items()
                        if key in metadata_fields
                    }

                    image_params.append(
                        {"data_id": frame["image_uuid"], "metadata": metadata}
                    )

                return [
                    result
                    for result in self._multi_thread(
                        self._get_rheed_image_result, kwargs_list=image_params
                    )
                    if result
                ]

            cluster_image_results = (
                __obtain_frame_data(
                    cluster_frames[0],
                    ["cluster_id", "start_timestamp_seconds", "end_timestamp_seconds"],
                )
                if cluster_frames
                else None
            )

            extracted_image_results = (
                __obtain_frame_data(
                    extracted_frames,
                    ["timestamp_seconds"],
                )
                if extracted_frames
                else None
            )
            return RHEEDVideoResult(
                data_id=data_id,
                timeseries_data=timeseries_data,
                cluster_image_data=cluster_image_results,
                snapshot_image_data=extracted_image_results,
                rotating=data_type == "rheed_rotating",
            )

        raise ValueError("Data type must be rheed_video, rheed_image, or xps")

    def _get_rheed_timeseries_result(self, data_id: str):
        # Get rheed video timeseries results
        plot_data = self._get(sub_url=f"clusters/plot_data/{data_id}")
        timeseries_data = DataFrame(plot_data)
        timeseries_data["cluster_std"] = (
            timeseries_data["cluster_mean_plus_std"] - timeseries_data["cluster_id"]
        )
        timeseries_data["first_order_intensity"] = (
            timeseries_data["region_1_intensity"]
            + timeseries_data["region_2_intensity"]
        ) / 2.0

        timeseries_data = timeseries_data.drop(
            columns=[
                "cluster_mean",
                "cluster_cdf",
                "region_1_intensity",
                "region_2_intensity",
                "cluster_mean_plus_std",
                "cluster_mean_minus_std",
            ]
        )

        column_mapping = {
            "real_time_seconds": "Time",
            "cluster_id": "Cluster ID",
            "cluster_std": "Cluster ID Uncertainty",
            "relative_strain": "Relative Strain",
            "cumulative_strain": "Cumulative Strain",
            "oscillation_period_seconds": "Oscillation Period",
            "spot_count": "Diffraction Spot Count",
            "first_order_intensity": "First Order Intensity",
            "region_0_intensity": "Specular Intensity",
        }

        return timeseries_data.rename(columns=column_mapping)

    def _get_rheed_image_result(self, data_id: str, metadata: dict | None = None):
        # Get pattern graph data
        if metadata is None:
            metadata = {}
        graph_data = self._get(sub_url=f"spots/{data_id}")
        graph = (
            nx.node_link_graph(graph_data, source="start_node", target="end_node")
            if graph_data
            else None
        )

        # Get raw and processed image data
        image_download: dict[str, str] = self._get(  # type: ignore  # noqa: PGH003
            sub_url=f"data_entries/processed_data/{data_id}",
            params={"return_as": "url-download"},
        )

        if image_download is None:
            return None

        # Image is pulled from the S3 pre-signed URL
        image_bytes: bytes = self._get(  # type: ignore  # noqa: PGH003
            base_override=image_download["url"], sub_url="", deserialize=False
        )

        image_data = Image.open(BytesIO(image_bytes))

        if graph is not None:
            min_node_index = min(graph.nodes(), key=lambda x: int(x))
            data_id = graph.nodes(data=True)[min_node_index]["uuid"]

        return RHEEDImageResult(
            data_id=data_id,
            processed_image=image_data,
            pattern_graph=graph,
            metadata=metadata,
        )

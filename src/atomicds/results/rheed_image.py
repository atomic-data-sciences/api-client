from __future__ import annotations

from uuid import UUID

import networkx as nx
import numpy as np
import pandas as pd
import trackpy as tp
from monty.json import MSONable
from networkx import Graph
from PIL import Image as PILImage
from PIL import ImageDraw
from PIL.Image import Image
from pycocotools import mask

from atomicds.core import ClientError, boxes_overlap

tp.quiet()


class RHEEDImageResult(MSONable):
    def __init__(
        self,
        data_id: UUID | str,
        processed_image: Image,
        pattern_graph: Graph | None,
        metadata: dict | None = None,
    ):
        """RHEED image result

        Args:
            data_id (UUID | str): Data ID for the entry in the data catalogue.
            processed_image (Image): Processed image data in a PIL Image format.
            pattern_graph (Graph | None): NetworkX Graph object for the extracted diffraction pattern.
            metadata (dict): Generic metadata (e.g. timestamp, cluster_id, etc...).
        """

        metadata = metadata or {}

        self.data_id = data_id
        self.processed_image = processed_image
        self.pattern_graph = pattern_graph
        self.metadata = metadata

    def get_plot(self, show_mask: bool = True, show_spot_nodes: bool = True) -> Image:
        """Get diffraction pattern image with optional overlays

        Args:
            show_mask (bool): Whether to show mask overlay of identified pattern. Defaults to True.
            show_spot_nodes (bool): Whether to show identified diffraction node overlays. Defaults to True.

        Returns:
            (Image): PIL Image object with optional overlays

        """
        image = self.processed_image.copy().convert("RGBA")
        draw = ImageDraw.Draw(image)
        if self.pattern_graph:
            masks = []
            for _, node_data in self.pattern_graph.nodes.data():
                if show_mask:
                    mask_rle = node_data.get("mask_rle")
                    mask_width = node_data.get("mask_width")
                    mask_height = node_data.get("mask_height")

                    if mask_rle and mask_width and mask_height:
                        mask_dict = {
                            "counts": mask_rle,
                            "size": (mask_height, mask_width),
                        }
                        masks.append(mask.decode(mask_dict))  # type: ignore  # noqa: PGH003

                if show_spot_nodes:
                    # Draw nodes
                    y = node_data.get("centroid_0")
                    x = node_data.get("centroid_1")

                    if x and y:
                        center = (x, y)
                        radius = 0.02 * max(image.width, image.height)
                        color = (255, 0, 0, 255)

                        draw.ellipse(
                            (
                                center[0] - radius,
                                center[1] - radius,
                                center[0] + radius,
                                center[1] + radius,
                            ),
                            fill=color,
                        )

            if show_mask:
                total_mask = np.stack(masks, axis=0).sum(axis=0).squeeze()
                overlay = np.zeros((*total_mask.shape, 4), dtype=np.uint8)
                overlay[np.where(total_mask)] = [255, 0, 0, int(0.2 * (255))]

                overlay = PILImage.fromarray(overlay)
                image.paste(overlay, mask=overlay)

        return image

    def get_laue_zero_radius(self) -> tuple[float, tuple[float, float]]:
        """
        Get the radius of the zeroth order Laue zone. Note that the data is symmetrized across the
        vertical axis before the Laue zone is searched for.

        Returns:
            (tuple[float, tuple[float, float]]): Tuple containing the best fit radius and center point.
        """

        node_data = []
        if self.pattern_graph is not None:
            node_data.append(
                pd.DataFrame.from_dict(
                    dict(self.pattern_graph.nodes(data=True)), orient="index"
                )
            )

        node_df = pd.concat(node_data, axis=0).reset_index(drop=True)
        node_df = self._symmetrize(node_df)

        image_array = np.array(self.processed_image)

        thetas = np.linspace(0, 2 * np.pi, 1440)
        radiis = np.linspace(100, 1000, 100)

        intersection_point = (
            node_df.loc[node_df["node_id"] == node_df["node_id"].min()][
                ["centroid_0", "centroid_1"]
            ]
            .to_numpy()
            .squeeze()
        )
        result_matrix = []

        for r0 in radiis:
            x_center, y_center = intersection_point[1], intersection_point[0] - r0
            x_points = r0 * np.cos(thetas) + x_center
            y_points = r0 * np.sin(thetas) + y_center
            int_x_points = np.round(x_points).astype(int)
            int_y_points = np.round(y_points).astype(int)
            coords = np.hstack(
                [int_x_points.reshape(-1, 1), int_y_points.reshape(-1, 1)]
            )
            coords = coords[coords[:, 1] >= 0]
            coords = coords[coords[:, 0] >= 0]
            coords = coords[coords[:, 1] < image_array.shape[0]]
            coords = coords[coords[:, 0] < image_array.shape[1]]
            if len(coords) == 0:
                continue
            intensities = image_array[coords[:, 1], coords[:, 0]]
            result_matrix.append(np.quantile(intensities, 0.9))

        best_fit_radius = np.mean(radiis[np.argsort(result_matrix)][-3:])

        return best_fit_radius, (intersection_point[0] - best_fit_radius, x_center)  # type: ignore  # noqa: PGH003

    def get_pattern_dataframe(
        self,
        extra_data: dict | None = None,
        symmetrize: bool = False,
    ) -> pd.DataFrame:
        """Featurize the RHEED image collection into a dataframe of node features and edge features.

        Args:
            extra_data (dict | None): Dictionary containing field names and values of extra data to be included in the DataFrame object.
                Defaults to None.
            fields_to_retain (list[str] | None): Fields to ensure are kept in the DataFrame object. Defaults to None which
            symmetrize (bool): Whether to symmetrize the data across the vertical axis. Defaults to False.

        Returns:
            (DataFrame): Pandas DataFrame object of RHEED node and edge features.
        """

        extra_data = extra_data or {}

        node_feature_cols = [
            "spot_area",
            "streak_area",
            "relative_centroid_0",
            "relative_centroid_1",
            "intensity_centroid_0",
            "intensity_centroid_1",
            "specular_origin_0",
            "specular_origin_1",
            "fwhm_0",
            "fwhm_1",
            "area",
            "eccentricity",
            "center_distance",
            "axis_major_length",
            "axis_minor_length",
        ]

        # TODO: add edge features
        # edge_feature_cols = ["weight", "horizontal_weight", "vertical_weight", "horizontal_overlap"]

        node_data = []
        if self.pattern_graph is not None:
            node_data.append(
                pd.DataFrame.from_dict(
                    dict(self.pattern_graph.nodes(data=True)), orient="index"
                )
            )

        node_df = pd.concat(node_data, axis=0).reset_index(drop=True)

        if symmetrize:
            node_df = self._symmetrize(node_df)

        extra_data_df = pd.DataFrame.from_records(
            [{"data_id": self.data_id} | extra_data]
        )
        feature_df: pd.DataFrame = node_df.pivot_table(
            index="uuid", columns="node_id", values=node_feature_cols
        )

        feature_df.columns = feature_df.columns.to_flat_index()

        feature_df = feature_df.merge(
            extra_data_df, left_index=True, right_on="data_id", how="inner"
        )
        feature_df = feature_df.rename(
            columns={col: (col, "") for col in extra_data_df.columns}
        )
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns)

        keep_cols = node_feature_cols + list(extra_data.keys())

        return feature_df[keep_cols]  # type: ignore  # noqa: PGH003

    @staticmethod
    def _symmetrize(node_df: pd.DataFrame):
        """Symmetrize a DataFrame object containing RHEED image node data"""

        reflection_plane = node_df["specular_origin_1"].mean()

        left_nodes = node_df.loc[node_df["centroid_1"] < reflection_plane]
        right_nodes = node_df.loc[node_df["centroid_1"] > reflection_plane]

        left_to_right = left_nodes.copy()
        left_to_right["centroid_1"] = reflection_plane + (
            reflection_plane - left_to_right["centroid_1"]
        )
        left_to_right["intensity_centroid_1"] = -left_to_right["intensity_centroid_1"]
        left_to_right["relative_centroid_1"] = -left_to_right["relative_centroid_1"]

        new_max = (
            reflection_plane + (reflection_plane - left_to_right["bbox_minc"])
        ).astype(int)
        new_min = (
            reflection_plane + (reflection_plane - left_to_right["bbox_maxc"])
        ).astype(int)
        left_to_right["bbox_maxc"] = new_max
        left_to_right["bbox_minc"] = new_min

        left_to_right["node_id"] = left_to_right["node_id"] + 1000

        right_to_left = right_nodes.copy()
        right_to_left["centroid_1"] = reflection_plane - (
            right_to_left["centroid_1"] - reflection_plane
        )
        right_to_left["intensity_centroid_1"] = -right_to_left["intensity_centroid_1"]
        right_to_left["relative_centroid_1"] = -right_to_left["relative_centroid_1"]

        new_max = (
            reflection_plane - (right_to_left["bbox_minc"] - reflection_plane)
        ).astype(int)
        new_min = (
            reflection_plane - (right_to_left["bbox_maxc"] - reflection_plane)
        ).astype(int)

        right_to_left["bbox_minc"] = new_min
        right_to_left["bbox_maxc"] = new_max
        right_to_left["node_id"] = right_to_left["node_id"] + 1000

        node_df = pd.concat(
            [node_df, left_to_right, right_to_left], axis=0
        ).reset_index(drop=True)

        if node_df.empty:
            return node_df

        first_row = node_df.iloc[[0]].pop("last_updated")
        original_dtypes = first_row.dtypes

        # merge rows with overlapping bounding boxes into one row
        drop_rows = []
        add_rows = []
        for i in range(
            len(node_df) - 1
        ):  # -1 so we don't try to pair the last row with anything
            for j in range(
                i + 1, len(node_df)
            ):  # Start from i+1 to avoid repeats and self-pairing
                row1 = node_df.iloc[[i]]
                row2 = node_df.iloc[[j]]
                if boxes_overlap(
                    row1[["bbox_minc", "bbox_minr", "bbox_maxc", "bbox_maxr"]]
                    .to_numpy()
                    .squeeze()
                    .tolist(),
                    row2[["bbox_minc", "bbox_minr", "bbox_maxc", "bbox_maxr"]]
                    .to_numpy()
                    .squeeze()
                    .tolist(),
                ):
                    merged_row = (
                        pd.concat([row1, row2], axis=0)
                        .sort_values("node_id")
                        .reset_index(drop=True)
                    )
                    merged_row = merged_row.agg(
                        {
                            "centroid_0": "mean",
                            "centroid_1": "mean",
                            "intensity_centroid_0": "mean",
                            "intensity_centroid_1": "mean",
                            "relative_centroid_0": "mean",
                            "relative_centroid_1": "mean",
                            "bbox_minc": "mean",
                            "bbox_minr": "mean",
                            "bbox_maxc": "mean",
                            "bbox_maxr": "mean",
                            "area": "mean",
                            "node_id": "min",
                            "pattern_id": lambda x: x.iloc[0],
                            "specular_origin_0": "mean",
                            "specular_origin_1": "mean",
                            "center_distance": "mean",
                            "fwhm_0": "mean",
                            "fwhm_1": "mean",
                            "mask_rle": lambda x: x.iloc[0],
                            "mask_width": lambda x: x.iloc[0],
                            "mask_height": lambda x: x.iloc[0],
                            "uuid": lambda x: x.iloc[0],
                            "oscillation_period_seconds": "mean",
                            "eccentricity": "mean",
                            "axis_major_length": "mean",
                            "axis_minor_length": "mean",
                            "bbox_intensity": "mean",
                            "spot_area": "mean",
                            "streak_area": "mean",
                        }
                    )

                    drop_rows.append(i)
                    drop_rows.append(j)
                    add_rows.append(merged_row)

        node_df = node_df.drop(drop_rows, axis=0)
        merged_df = pd.concat(add_rows, axis=1).T.astype(original_dtypes)
        return pd.concat([node_df, merged_df], axis=0).reset_index(drop=True)


# TODO: Add tests for RHEEDImageCollection
class RHEEDImageCollection(MSONable):
    def __init__(
        self, rheed_images: list[RHEEDImageResult], extra_data: list[dict] | None = None
    ):
        """Collection of RHEED images

        Args:
            rheed_images (list[RHEEDImageResult]): List of RHEEDImageResult objects.
            extra_data (list[dict] | None): List of dictionaries containing field names and values of extra data to be included in the DataFrame object.
                Defaults to None.
        """

        extra_data = extra_data or []  # type: ignore  # noqa: PGH003

        if len(extra_data) > 0 and len(extra_data) != len(rheed_images):
            raise ValueError(
                "List of extra data must be the same length as the RHEED image collection."
            )

        for idx, rheed_image in enumerate(rheed_images):
            if rheed_image.pattern_graph:
                for node in rheed_image.pattern_graph.nodes:
                    rheed_image.pattern_graph.nodes[node]["pattern_id"] = idx

        self.rheed_images = rheed_images
        self.extra_data = extra_data

    def align_fingerprints(self) -> tuple[pd.DataFrame, list[RHEEDImageResult]]:
        """
        Align a collection of RHEED fingerprints by relabeling the nodes to connect the same scattering
        features across RHEED patterns, based on relative position to the center feature.

        Returns:
            (tuple[DataFrame, list[RHEEDImageResult]): Pandas DataFrame object with aligned RHEED fingerprint data
        """
        image_scales = [
            rheed_image.processed_image.size for rheed_image in self.rheed_images
        ]
        image_scale = np.amax(image_scales, axis=0)
        data_ids = [rheed_image.data_id for rheed_image in self.rheed_images]

        node_data = []
        for ind, rheed_image in enumerate(self.rheed_images):
            if rheed_image.pattern_graph is None:
                raise ClientError(
                    f"Unable to align fingerprints as rheed image {ind} has no graph data."
                )
            node_data.append(
                pd.DataFrame.from_dict(
                    dict(rheed_image.pattern_graph.nodes(data=True)), orient="index"
                )
            )

        node_df = pd.concat(
            node_data,
            axis=0,
        ).reset_index(drop=True)

        labels, _ = pd.factorize(node_df["uuid"])
        node_df["pattern_id"] = labels

        linked_df = tp.link(
            f=node_df,
            search_range=np.sqrt(np.sum(np.square(image_scale))) * 0.05,
            memory=len(data_ids),
            t_column="pattern_id",
            pos_columns=["relative_centroid_1", "relative_centroid_0"],
        )

        rheed_images: list[RHEEDImageResult] = []
        splits = [group for _, group in linked_df.groupby("pattern_id")]
        for split, rheed_image in zip(splits, self.rheed_images):
            mapping = dict(zip(split["node_id"], split["particle"]))
            rheed_image.pattern_graph = nx.relabel_nodes(  # type: ignore  # noqa: PGH003
                rheed_image.pattern_graph,  # type: ignore  # noqa: PGH003
                mapping,
            )
            rheed_images.append(rheed_image)

        return linked_df, rheed_images

    def get_pattern_dataframe(
        self, streamline: bool = True, normalize: bool = True
    ) -> pd.DataFrame:
        """Featurize the RHEED image collection into a dataframe of node features and edge features.

        Args:
            streamline (bool): Whether to remove streamline the DataFrame object and remove null values. Defaults to True.
            normalize (bool): Whether to min/max normalize the feature data across all images. Defaults to True.

        Returns:
            (DataFrame): Pandas DataFrame object of RHEED node and edge features.
        """

        node_feature_cols = [
            "spot_area",
            "streak_area",
            "relative_centroid_0",
            "relative_centroid_1",
            "intensity_centroid_0",
            "intensity_centroid_1",
            "specular_origin_0",
            "specular_origin_1",
            "fwhm_0",
            "fwhm_1",
            "area",
            "eccentricity",
            "center_distance",
            "axis_major_length",
            "axis_minor_length",
        ]

        # TODO: add edge features
        # edge_feature_cols = ["weight", "horizontal_weight", "vertical_weight", "horizontal_overlap"]

        feature_dfs = [
            rheed_image.get_pattern_dataframe(extra_data=extra_data)
            for rheed_image, extra_data in zip(self.rheed_images, self.extra_data)
        ]

        feature_df = pd.concat(feature_dfs, axis=0).reset_index(drop=True)

        keep_cols = node_feature_cols + list(
            {key for extra_data in self.extra_data for key in extra_data}
        )

        feature_df = feature_df[keep_cols]

        if streamline:
            feature_df = feature_df.dropna(axis=1)

        if normalize:
            # min max normalization
            feature_df = (feature_df - feature_df.min()) / (
                feature_df.max() - feature_df.min()
            )

        return feature_df  # type: ignore  # noqa: PGH003

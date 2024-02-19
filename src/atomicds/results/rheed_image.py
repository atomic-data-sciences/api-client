from typing import Optional
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

tp.quiet()


class RHEEDImageResult(MSONable):
    def __init__(
        self,
        data_id: UUID | str,
        processed_image: Image,
        # parent_data_id: UUID | str | None,
        pattern_graph: Graph | None,
        metadata: Optional[dict] = None,
        labels: Optional[dict] = None
    ):
        """RHEED image result

        Args:
            data_id (UUID | str): Data ID for the entry in the data catalogue.
            processed_image (Image): Processed image data in a PIL Image format.
            pattern_graph (Graph | None): NetworkX Graph object for the extracted diffraction pattern.
            metadata (dict): Generic metadata (e.g. timestamp, cluster_id, etc...).
        """
        if labels is None:
            labels = {}
        if metadata is None:
            metadata = {}

        self.data_id = data_id
        # self.parent_data_id = parent_data_id
        self.processed_image = processed_image
        self.pattern_graph = pattern_graph
        self.metadata = metadata
        self.labels = labels

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


class RHEEDImageCollection(MSONable):

    def __init__(self, rheed_images: list[RHEEDImageResult], labels: Optional[list[dict]] = None):
        """Collection of RHEED images

        Args:
            rheed_images (list[RHEEDImageResult]): List of RHEEDImageResult objects.
        """
        if labels is None:
            labels = {}
        if len(labels) > 0 and len(labels) != len(rheed_images):
            raise ValueError("Labels must be the same length as the RHEED image collection.")

        # if labels are provided, add an ordering into pattern_id
        for idx, (rheed_image, label) in enumerate(zip(rheed_images, labels)):
            rheed_image.labels = rheed_image.labels | label
            for node in rheed_image.pattern_graph.nodes:
                rheed_image.pattern_graph.nodes[node]["pattern_id"] = idx


        self.rheed_images = rheed_images


    def align_fingerprints(self):
        """Align a collection of RHEED fingerprints by relabeling the nodes to connect the same scattering features across RHEED patterns, based on relative position to the center feature.
        """
        image_scales = [rheed_image.processed_image.size for rheed_image in self.rheed_images]
        image_scale = np.amax(image_scales, axis=0)
        data_ids = [rheed_image.data_id for rheed_image in self.rheed_images]

        node_df = pd.concat(
            [
                pd.DataFrame.from_dict(dict(rheed_image.pattern_graph.nodes(data=True)), orient='index')
                for rheed_image in self.rheed_images
            ],
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

        rheed_images = []
        splits = [group for _, group in linked_df.groupby("pattern_id")]
        for split, rheed_image in zip(splits, self.rheed_images):
            mapping = dict(zip(split["node_id"], split["particle"]))
            rheed_image.pattern_graph = nx.relabel_nodes(rheed_image.pattern_graph, mapping)
            rheed_images.append(rheed_image)

        self.rheed_images = rheed_images

        return linked_df


    def featurize(self, streamline: bool = True, normalize: bool = True, **kwargs):
        """Featurize the RHEED image collection into a dataframe of node features and edge features.

        Returns:
            (pd.DataFrame): DataFrame of node features.
            (pd.DataFrame): DataFrame of edge features.
        """

        node_feature_cols = [
            "spot_area",
            "streak_area",
            "relative_centroid_0",
            "relative_centroid_1",
            "intensity_centroid_0",
            "intensity_centroid_1",
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

        node_df = pd.concat(
            [
                pd.DataFrame.from_dict(dict(rheed_image.pattern_graph.nodes(data=True)), orient='index')
                for rheed_image in self.rheed_images
            ],
            axis=0,
        ).reset_index(drop=True)

        label_df = pd.DataFrame.from_records([{"data_id": rheed_image.data_id} | rheed_image.labels for rheed_image in self.rheed_images])
        feature_df = node_df.pivot_table(
            index="uuid", columns="node_id", values=node_feature_cols
        )

        feature_df.columns = feature_df.columns.to_flat_index()
        feature_df = feature_df.merge(label_df, left_index=True, right_on="data_id", how="inner")
        feature_df = feature_df.rename(columns={col: (col, "") for col in label_df.columns})
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns)

        keep_cols = node_feature_cols + list({key for rheed_image in self.rheed_images for key in rheed_image.labels})
        feature_df = feature_df[keep_cols]

        if streamline:
            feature_df = feature_df.dropna(axis=1)

        if normalize:
            # min max normalization
            feature_df = (feature_df - feature_df.min()) / (feature_df.max() - feature_df.min())

        self.feature_df = feature_df

        return feature_df

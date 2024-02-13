from uuid import UUID

from monty.json import MSONable
from networkx import Graph
from PIL.Image import Image
import trackpy as tp

# tp.quiet()

class RHEEDImageResult(MSONable):
    def __init__(
        self,
        data_id: UUID | str,
        processed_image: Image,
        pattern_graph: Graph,
        metadata: dict | None = None,
    ):
        """RHEED image result

        Args:
            data_id (UUID | str): Data ID for the entry in the data catalogue.
            processed_image (Image): Processed image data in a PIL Image format.
            pattern_graph (Graph): NetworkX Graph object for the extracted diffraction pattern.
            metadata (dict): Generic metadata (e.g. timestamp, cluster_id, etc...).
        """
        self.data_id = data_id
        self.processed_image = processed_image
        self.pattern_graph = pattern_graph
        self.metadata = metadata

    def get_plot(self):
        # TODO: Implement
        pass


class RHEEDImageCollection(MSONable):
    def __init__(self, rheed_images: list[RHEEDImageResult]):
        """Collection of RHEED images

        Args:
            rheed_images (list[RHEEDImageResult]): List of RHEEDImageResult objects.
        """
        self.rheed_images = rheed_images

    def align_images(self, as_features: bool = False):

        image_scales = [rheed_image.processed_image.size for rheed_image in self.rheed_images]
        image_scale = np.amax(image_scales, axis=0)

        node_df = pd.concat(
            [
                rheed_image.pattern_graph.nodes(data=True)
                for rheed_image in self.rheed_images
            ],
            axis=0,
        ).reset_index(drop=True)

        print(node_df)

        labels, _ = pd.factorize(node_df["uuid"])
        node_df["pattern_id"] = labels

        linked_df = tp.link(
            f=node_df,
            search_range=np.sqrt(np.sum(np.square(image_scale))) * 0.05,
            memory=len(data_ids),
            t_column="pattern_id",
            pos_columns=["relative_centroid_1", "relative_centroid_0"],
        )

        adjacency_df_list = []
        node_df_list = []
        for pattern_id, data_id in linked_df[["pattern_id", "uuid"]].drop_duplicates().values:
            pid_subset = linked_df.loc[linked_df["pattern_id"] == pattern_id]
            node_df_subset = node_df.loc[node_df["pattern_id"] == pattern_id].copy()
            node_df_subset["node_id"] = node_df_subset["node_id"].replace(
                {n: p for (n, p) in zip(pid_subset["node_id"].values, pid_subset["particle"].values)}
            )
            node_df_list.append(node_df_subset)
            adjacency_df = RheedPattern._regions_as_graph(node_df_subset, pattern_id, data_id)
            adjacency_df_list.append(adjacency_df)

        node_df = pd.concat(node_df_list, axis=0).reset_index(drop=True)
        adjacency_df = pd.concat(adjacency_df_list, axis=0).reset_index(drop=True)

        if as_features:
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
            edge_feature_cols = ["weight", "horizontal_weight", "vertical_weight", "horizontal_overlap"]

            node_df = node_df.dataframe.pivot(
                index="pattern_id", columns="node_id", values=node_feature_cols
            )
            adjacency_df = adjacency_df.dataframe.pivot(
                index="pattern_id", columns=["start_node", "end_node"], values=edge_feature_cols
            )
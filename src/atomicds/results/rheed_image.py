from uuid import UUID

from monty.json import MSONable
from networkx import Graph
from PIL.Image import Image


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

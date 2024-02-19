from uuid import UUID

import numpy as np
from monty.json import MSONable
from networkx import Graph
from PIL import Image as PILImage
from PIL import ImageDraw
from PIL.Image import Image
from pycocotools import mask


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

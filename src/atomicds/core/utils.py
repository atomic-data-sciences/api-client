import numpy as np
import numpy.typing as npt


def boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap

    Args:
        box1 (list[float]): List of xmin, ymin, xmax, ymax coordinates defining first box
        box2 (list[float]): List of xmin, ymin, xmax, ymax coordinates defining second box

    Returns:
        (bool): True if the boxes overlap
    """
    # Unpack coordinates
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    # Check for overlap
    if xmax1 < xmin2 or xmax2 < xmin1 or ymax1 < ymin2 or ymax2 < ymin1:
        return False
    return True


def convert_to_polar_coordinates(points: npt.NDArray, origin=(0, 0), scale=1.0):
    """
    Convert a set of 2D points to polar coordinates with radius and angle.

    Args:
        points (NDArray): Numpy array containing a list of points.
        origin (tuple[int, int]): Origin point.
        scale (float): Scaling number.
    """

    # Calculate the relative position of the points to the origin
    relative_points = points - origin

    # Calculate the radius and angle of the points
    intermediate = np.sum(np.square(relative_points), axis=1)
    radius = np.sqrt(intermediate) / scale
    angle = np.arctan2(relative_points[:, 1], relative_points[:, 0])

    # Stack the radius and angle into a single array
    return np.stack([radius, angle], axis=1)

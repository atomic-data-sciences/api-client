import numpy as np
import numpy.typing as npt


def normalize_pixel_dimensions(
    points: npt.NDArray, image_shape: tuple[int, int]
) -> npt.NDArray:
    """
    Rescale pixel dimensions to a new image shape.

    Args:
        points (NDArray): Numpy array containing a list of points with columns width, height.
        image_shape (tuple[int, int]): Image shape.

    Returns:
        NDArray: Numpy array containing the rescaled points.
    """
    height, width = image_shape

    points[:, 0] = points[:, 0] / width
    points[:, 1] = points[:, 1] / height

    return points


def boxes_overlap(box1, box2) -> bool:
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


def rescale_cartesian_coordinates(
    points: npt.NDArray, origin=(0, 0), scale: float = 1.0
) -> npt.NDArray:
    """
    Normalize radius in polar coordinates, then convert back to cartesian to get rescaled cartesian coordinates in image dimensions.
    Args:
        points (NDArray): Numpy array containing a list of points.
        origin (tuple[int, int]): Origin point.
        scale (float): Scaling number.
    Returns:
        NDArray: Numpy array containing the rescaled points.
    """

    # Convert the points to polar coordinates
    polar_coordinates = convert_to_polar_coordinates(points, origin=origin, scale=scale)

    scaled_1 = polar_coordinates[:, 0] * np.cos(polar_coordinates[:, 1])
    scaled_0 = polar_coordinates[:, 0] * np.sin(polar_coordinates[:, 1])

    return np.stack([scaled_0, scaled_1], axis=1)


def convert_to_polar_coordinates(
    points: npt.NDArray, origin=(0, 0), scale=1.0
) -> npt.NDArray:
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

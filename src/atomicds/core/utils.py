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

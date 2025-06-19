import numpy as np
from scipy.ndimage import uniform_filter, binary_opening, label, sum as ndi_sum

def smooth_mask(mask, size=5):
    return uniform_filter(mask.astype(float), size=size) > 0.5

def remove_isolated_pixels(mask, structure=np.ones((3, 3))):
    return binary_opening(mask, structure=structure)

def remove_small_objects(mask, min_size=10):
    labeled_array, num_features = label(mask)
    sizes = np.bincount(labeled_array.ravel())
    mask_sizes = sizes < min_size
    mask_sizes[0] = False
    return mask_sizes[labeled_array]

def remove_small_clusters(kmeans_labels, min_size=5):
    cleaned_labels = kmeans_labels.copy()
    unique_classes = np.unique(kmeans_labels[kmeans_labels > 0])

    for class_id in unique_classes:
        binary_mask = (kmeans_labels == class_id)
        labeled_array, num_features = label(binary_mask)
        region_sizes = ndi_sum(binary_mask, labeled_array, index=range(1, num_features + 1))

        for region_id, size in enumerate(region_sizes, start=1):
            if size < min_size:
                mask = labeled_array == region_id
                neighbor_labels = kmeans_labels[np.logical_and(~mask, np.roll(mask, 1, axis=0))]
                neighbor_labels = np.append(neighbor_labels, kmeans_labels[np.logical_and(~mask, np.roll(mask, -1, axis=0))])
                neighbor_labels = np.append(neighbor_labels, kmeans_labels[np.logical_and(~mask, np.roll(mask, 1, axis=1))])
                neighbor_labels = np.append(neighbor_labels, kmeans_labels[np.logical_and(~mask, np.roll(mask, -1, axis=1))])
                neighbor_labels = neighbor_labels[neighbor_labels > 0]
                if len(neighbor_labels) > 0:
                    new_class = np.bincount(neighbor_labels).argmax()
                    cleaned_labels[mask] = new_class
    return cleaned_labels

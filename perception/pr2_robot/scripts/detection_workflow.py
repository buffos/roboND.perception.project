#!/usr/bin/env python
"""Segmentation and Detection Workflow"""

from pcl_helper import *


def statistical_filtering(pcl_data):
    """Get point cloud data and return
    filtered  data for statistical outliers"""

    outlier_filter = pcl.StatisticalOutlierRemovalFilter_PointXYZRGB(pcl_data)
    # n of neighboring points to analyze
    outlier_filter.set_mean_k(30)
    # any point with distance larger than mean + 0.3 * std
    # is considered an outlier
    outlier_filter.set_std_dev_mul_thresh(0.3)
    filtered_data = outlier_filter.filter()

    return filtered_data


def voxel_downsampling(pcl_data, leaf_size=0.01):
    """Downsampling data to reduce point cloud size"""
    vox = pcl_data.make_voxel_grid_filter()
    # leaf_size == voxel size
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
    filtered_data = vox.filter()  # apply filter to the point cloud.

    return filtered_data


def passthrough_filtering(pcl_data, filter_axis='z',
                          min_value=0.0, max_value=1.0):
    """Filter data along axis, considering points outside min max_values
    as outliers"""

    passthrough = pcl_data.make_passthrough_filter()
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(min_value, max_value)
    filtered_data = passthrough.filter()

    return filtered_data


def ransac_plane_segmentation(pcl_data, plane_height=0.01):
    """ Fit points around a plane of given height.
    Inliers = the plane (table)
    Ouliers = the objects on or below the table
    """
    seg = pcl_data.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)  # Set the model you wish to fit
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(plane_height)
    inliers, coefficients = seg.segment()
    # Extract inliers and outliers
    pcl_inliers = pcl_data.extract(inliers, negative=False)
    pcl_outliers = pcl_data.extract(inliers, negative=True)

    return pcl_inliers, pcl_outliers


def euclidean_clustering(pcl_data, tolerance=0.025, min_size=20, max_size=300):
    """ Cluster the point cloud into segments based on euclidean distance"""

    #  a. construct a k-d tree.
    white_cloud = XYZRGB_to_XYZ(pcl_data)  # XYZRGB point cloud to XYZ
    tree = white_cloud.make_kdtree()

    #  b. create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    #  c. set tolerances for distance threshold
    #     as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(tolerance)
    ec.set_MinClusterSize(min_size)
    ec.set_MaxClusterSize(max_size)
    #  e. search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    #  f. extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    return cluster_indices


def colorize_cluster(cluster_data, cluster_indices):
    """Get a point cloud of data that was already clustered
    and also get cluster_indices, with each indice array containg
    the points from that object. Then attach to the coordinates of
    each point a color, one for each object and convert back to XYZRGB
    """
    # get again the data in XYZ format
    white_cloud = XYZRGB_to_XYZ(cluster_data)
    #  a. assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    #  b. for each point in each cluster append it to the
    #     color_cluster_point_list as XYZRGB list
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #  c. create a new cloud containing all clusters each with a unique color
    cluster_out = pcl.PointCloud_PointXYZRGB()
    cluster_out.from_list(color_cluster_point_list)

    return cluster_out

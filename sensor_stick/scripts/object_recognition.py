#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # 1: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # 2: Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.01 # selecting voxel size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter() # apply filter to the point cloud.

    # 3: PassThrough Filter
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    # 4: RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)  # Set the model you wish to fit
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # 5: Extract inliers and outliers
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)

    # 6: Euclidean Clustering
    #  a. construct a k-d tree.
    white_cloud = XYZRGB_to_XYZ(cloud_objects) # XYZRGB point cloud to XYZ
    tree = white_cloud.make_kdtree()

    #  b. create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    #  c. set tolerances for distance threshold
    #     as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.015)
    ec.set_MinClusterSize(20)
    ec.set_MaxClusterSize(1500)
    #  e. search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    #  f. extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # 7: Create Cluster-Mask Point Cloud to visualize each cluster separately
    #  a. assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))
    #  b. for each point in each cluster append it to the
    #     color_cluster_point_list as XYZRGB list
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #  c. create a new cloud containing all clusters each with a unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # 8: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # 9: Publish ROS messages
    # pcl_objects_pub.publish(ros_cloud_objects)
    # pcl_table_pub.publish(ros_cloud_table)
    # pcl_cluster_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    # for each of the segmented cluster
    for index, pts_list in enumerate(cluster_indices):
        # 1. Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        # 2. Convert the cluster from pcl to ROS
        pcl_cluster = pcl_to_ros(pcl_cluster)
        # 3. Extract color histogram features
        color_hists = compute_color_histograms(pcl_cluster, using_hsv=True)
        # 4. Extract normal histogram features
        normals = get_normals(pcl_cluster)
        normal_hists = compute_normal_histograms(normals)
        # 5. Compute the concatenated feature vector
        feature = np.concatenate((color_hists, normal_hists))
        detected_objects.append([feature])
        # 6. Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        # 7. Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))
        # 8. Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster_cloud
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

if __name__ == '__main__':

    # 1. ROS node initialization
    rospy.init_node('object_markers_pub', anonymous=True)

    # 2. Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)

    # 3. Create Publishers
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # 4. Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # 5. Initialize color_list
    get_color_list.color_list = []

    # 6. Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()

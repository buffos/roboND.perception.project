#!/usr/bin/env python

# Import modules
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

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

from detection_workflow import *


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Exercise-2 pipeline:
    # 1: Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)
    # 2. Statistical Outlier Filtering
    pcl_data = statistical_filtering(pcl_data)
    # 3: Voxel Grid Downsampling
    pcl_data = voxel_downsampling(pcl_data, leaf_size=0.005)
    # 4: PassThrough Filter
    # 4a: filter out the floor
    pcl_data = passthrough_filtering(pcl_data, filter_axis='z',
                                     min_value=0.6, max_value=2.0)
    # 4b: filter out  drop boxes
    pcl_data = passthrough_filtering(pcl_data, filter_axis='y',
                                     min_value=-0.5, max_value=0.5)

    # 5: Create and Publish the collision map
    #          publish a point cloud to `/pr2/3D_map/points`.
    collision_map = pcl_to_ros(pcl_data)
    collision_map_pub.publish(collision_map)

    # 6: RANSAC Plane Segmentation (table, objects)
    pcl_inliers, pcl_outliers = ransac_plane_segmentation(pcl_data, plane_height=0.01)


    # 7: Euclidean Clustering (outliers == objects)
    cluster_indices = euclidean_clustering(pcl_outliers, tolerance=0.015,
                                           min_size=20, max_size=3000)

    # 8: Create Cluster-Mask Point Cloud to visualize each cluster separately
    white_cloud = XYZRGB_to_XYZ(pcl_outliers)
    colored_cloud = colorize_cluster(pcl_outliers, cluster_indices)

    # 9: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(pcl_outliers)
    ros_cloud_table = pcl_to_ros(pcl_inliers)
    ros_cluster_cloud = pcl_to_ros(colored_cloud)

    # 10. Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Exercise-3 pipeline:
    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    # for each of the segmented cluster
    for index, pts_list in enumerate(cluster_indices):
        # 1. Grab the points for the cluster (pcl_outliers == cloud objects)
        pcl_for_object = pcl_outliers.extract(pts_list)
        # 2. Convert the cluster from pcl to ROS
        ros_for_object = pcl_to_ros(pcl_for_object)
        # 3. Extract color histogram features
        color_hists = compute_color_histograms(ros_for_object, using_hsv=True)
        # 4. Extract normal histogram features
        normals = get_normals(ros_for_object)
        normal_hists = compute_normal_histograms(normals)
        # 5. Compute the concatenated feature vector
        feature = np.concatenate((color_hists, normal_hists))
        # 6. Make the prediction and get the label
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
        do.cloud = ros_for_object
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels),
                                                       detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()

    pick_list = rospy.get_param('/object_list')
    print("Objects to Pick:  ",len(pick_list))
    print("Objects Detected: ",len(detected_objects_labels))
    num_detected = 0
    for item in pick_list:
        if item['name'] in detected_objects_labels:
            num_detected += 1

    print("Detected {0} / {1}: ".format(num_detected, len(pick_list)))
    # maybe here create a state machine to repeat detection if not optimal

    # a dictionary to be easier to find object by label
    detected_objects_list = dict(zip(detected_objects_labels, detected_objects))

    try:
        pr2_mover(detected_objects_list)
    except rospy.ROSInterruptException:
        pass

def create_pick_and_place_request(object_list):
    """Create the yaml file to send for the pick and place request
        Args:
             object_list: dictionary["label"] = detected_object_cloud
    """
    # 1. Initialize Variables
    test_scene_num = Int32()
    arm_name = String()
    object_name = String()
    object_group = String()
    pick_pose = Pose()
    place_pose = Pose()
    dictionaries_created = 0
    yaml_dicts = []

    # 1. read parameters
    pick_list = rospy.get_param('/object_list')
    test_scene_num.data = rospy.get_param('/test_scene_num')
    dropbox_params = rospy.get_param('/dropbox') # (position, name)


    for obj in pick_list:
        group = obj['group']
        name = obj['name']
        detected_obj = object_list.get(name) # if not found then None

        if detected_obj is not None:
            object_name.data = name
            object_group.data = group
            # calculate centroid
            points_arr = ros_to_pcl(detected_obj.cloud).to_array()
            centroid = np.mean(points_arr, axis=0)[:3]
            # define pick pose
            pick_pose.position.x = np.asscalar(centroid[0])
            pick_pose.position.y = np.asscalar(centroid[1])
            pick_pose.position.z = np.asscalar(centroid[2])
            pick_pose.orientation.x = 0.0
            pick_pose.orientation.y = 0.0
            pick_pose.orientation.z = 0.0
            pick_pose.orientation.w = 0.0

            # now create the place pick_pose
            for dropbox in dropbox_params:
                if group == dropbox['group']:
                    place_centroid = dropbox["position"]
                    place_pose.position.x = float(place_centroid[0]) + 0.1 * np.random.random()
                    place_pose.position.y = float(place_centroid[1]) + 0.1 * np.random.random()
                    place_pose.position.z = float(place_centroid[2])
                    place_pose.orientation.x = 0.0
                    place_pose.orientation.y = 0.0
                    place_pose.orientation.z = 0.0
                    place_pose.orientation.w = 0.0

                    # Assign the arm to be used for pick_place
                    arm_name.data = dropbox["name"]
                    print("Scene {}, picking up {} object, using {} arm, and placing it in the {} bin."
                                  .format(test_scene_num.data, object_name.data, arm_name.data, group))

                    # create the yaml dictionary
                    yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
                    yaml_dicts.append(yaml_dict)
                    dictionaries_created +=1
                    break;
            else:
                 print("Label: {} not found".format(name))

    return yaml_dicts


# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    yaml_dictionaries = create_pick_and_place_request(object_list)
    yaml_filename = "output_" + str(rospy.get_param('/test_scene_num')) + ".yaml"
    send_to_yaml(yaml_filename, yaml_dictionaries)

    # TODO: Initialize variables

    # TODO: Get/Read parameters

    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list

        # TODO: Get the PointCloud for a given object and obtain it's centroid

        # TODO: Create 'place_pose' for the object

        # TODO: Assign the arm to be used for pick_place

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

        # Wait for 'pick_place_routine' service to come up
        # rospy.wait_for_service('pick_place_routine')

        # try:
            # pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            # resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            # print ("Response: ",resp.success)

        # except rospy.ServiceException, e:
            # print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file



if __name__ == '__main__':

    # 1: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # 2: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # 3: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    collision_map_pub = rospy.Publisher("/pr2/3D_map/points", PointCloud2, queue_size=1)

    pub_world_joint_pub = rospy.Publisher('/pr2/world_joint_controller/command', Float64, queue_size=10)

    # 4: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # 5: Initialize color_list
    get_color_list.color_list = []

    # 6: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()

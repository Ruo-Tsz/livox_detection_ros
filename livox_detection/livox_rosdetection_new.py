#! /usr/bin/env python2
import os
import numpy as np
import tensorflow as tf
import copy
import config.config as cfg
from networks.model import *
import lib_cpp
import rospkg

import time

import rospy
import std_msgs.msg
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Quaternion
import sensor_msgs.point_cloud2 as pcl2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from std_msgs.msg import Int32MultiArray

mnum = 0
marker_array = MarkerArray()
marker_array_text = MarkerArray()

X_MAX = rospy.get_param("~max_x", 89.6)
X_MIN = rospy.get_param("~min_x", -89.6)
Y_MAX = rospy.get_param("~max_y", 49.4)
Y_MIN = rospy.get_param("~min_y", -49.4)
Z_MAX = rospy.get_param("~max_z", 3.0)
Z_MIN = rospy.get_param("~min_z", -3.0)
DX = DY = DZ = rospy.get_param("~voxel_size", 0.2)
overlap = rospy.get_param("~overlap", 11.2)
HEIGHT = int(round((X_MAX - X_MIN + 2 * overlap) / DX))
WIDTH = int(round((Y_MAX - Y_MIN) / DY))
CHANNELS = int(round((Z_MAX - Z_MIN) / DZ))

color_map = {
    "car": [1, 0, 0],
    "bus": [0.7, 0 ,0.7],
    "truck": [1, 0.7, 0.7],
    "pedestrian": [0, 1, 0],
    "bimo": [0, 0, 1]
} # cls: [r, g, b]


print(HEIGHT, WIDTH, CHANNELS)

T1 = np.array([[0.0, -1.0, 0.0, 0.0],
               [0.0, 0.0, -1.0, 0.0],
               [1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]]
              )
lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
         [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


class Detector(object):
    def __init__(self, nms_threshold=0.1, weight_file=None):
        self.net = livox_model(HEIGHT, WIDTH, CHANNELS)
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(cfg.GPU_INDEX)):
                input_bev_img_pl = \
                    self.net.placeholder_inputs(cfg.BATCH_SIZE)
                end_points = self.net.get_model(input_bev_img_pl)

                saver = tf.train.Saver()
                config = tf.ConfigProto()
                # config.gpu_options.allow_growth = True
                config.allow_soft_placement = True
                config.log_device_placement = False
                config.gpu_options.per_process_gpu_memory_fraction= 0.4
                self.sess = tf.Session(config=config)
                pkg_path = rospkg.RosPack().get_path('livox_detection')
                saver.restore(self.sess, pkg_path+"/model/livoxmodel")
                self.ops = {'input_bev_img_pl': input_bev_img_pl,  # input
                            'end_points': end_points,  # output
                            }
        rospy.init_node('livox_test', anonymous=True)
        self.sub = rospy.Subscriber("voxel", Int32MultiArray, queue_size=100, callback=self.LivoxCallback)
        self.marker_pub = rospy.Publisher('/detect_box3d', MarkerArray, queue_size=100)
        self.marker_text_pub = rospy.Publisher('/text_det', MarkerArray, queue_size=100)


    def roty(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    def get_3d_box(self, box_size, heading_angle, center):
        ''' Calculate 3D bounding box corners from its parameterization.

        Input:
            box_size: tuple of (l,w,h)
            heading_angle: rad scalar, clockwise from pos x axis
            center: tuple of (x,y,z)
        Output:
            corners_3d: numpy array of shape (8,3) for 3D box cornders
        '''
        R = self.roty(heading_angle)
        l, w, h = box_size
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + center[0]
        corners_3d[1, :] = corners_3d[1, :] + center[1]
        corners_3d[2, :] = corners_3d[2, :] + center[2]
        corners_3d = np.transpose(corners_3d)
        return corners_3d

    def data2voxel(self, pclist):

        data = [i*0 for i in range(HEIGHT*WIDTH*CHANNELS)]

        for line in pclist:
            X = float(line[0])
            Y = float(line[1])
            Z = float(line[2])
            if( Y > Y_MIN and Y < Y_MAX and
                X > X_MIN and X < X_MAX and
                Z > Z_MIN and Z < Z_MAX):
                channel = int((-Z + Z_MAX)/DZ)
                if abs(X)<3 and abs(Y)<3:
                    continue
                if (X > -overlap):
                    pixel_x = int((X - X_MIN + 2*overlap)/DX)
                    pixel_y = int((-Y + Y_MAX)/DY)
                    data[pixel_x*WIDTH*CHANNELS+pixel_y*CHANNELS+channel] = 1
                if (X < overlap):
                    pixel_x = int((-X + overlap)/DX)
                    pixel_y = int((Y + Y_MAX)/DY)
                    data[pixel_x*WIDTH*CHANNELS+pixel_y*CHANNELS+channel] = 1
        voxel = np.reshape(data, (HEIGHT, WIDTH, CHANNELS))
        return voxel

    def detect(self, batch_bev_img):
        feed_dict = {self.ops['input_bev_img_pl']: batch_bev_img}
        feature_out,\
            = self.sess.run([self.ops['end_points']['feature_out'],
                             ], feed_dict=feed_dict)
        result = lib_cpp.cal_result(feature_out[0,:,:,:], \
                                    cfg.BOX_THRESHOLD,overlap,X_MIN,HEIGHT, WIDTH, cfg.VOXEL_SIZE[0], cfg.VOXEL_SIZE[1], cfg.VOXEL_SIZE[2], cfg.NMS_THRESHOLD)
        is_obj_list = result[:, 0].tolist()
        
        reg_m_x_list = result[:, 5].tolist()
        reg_w_list = result[:, 4].tolist()
        reg_l_list = result[:, 3].tolist()
        obj_cls_list = result[:, 1].tolist()
        reg_m_y_list = result[:, 6].tolist()
        reg_theta_list = result[:, 2].tolist()
        reg_m_z_list = result[:, 8].tolist()
        reg_h_list = result[:, 7].tolist()

        results = []
        for i in range(len(is_obj_list)):
            box3d_pts_3d = lib_cpp.get_box_3d( \
                (reg_l_list[i], reg_w_list[i], reg_h_list[i]), \
                reg_theta_list[i], (reg_m_x_list[i], reg_m_y_list[i], reg_m_z_list[i]))
            if int(obj_cls_list[i]) == 0:
                cls_name = "car"
            elif int(obj_cls_list[i]) == 1:
                cls_name = "bus"
            elif int(obj_cls_list[i]) == 2:
                cls_name = "truck"
            elif int(obj_cls_list[i]) == 3:
                cls_name = "pedestrian"
            else:
                cls_name = "bimo"
            results.append([cls_name,
                            box3d_pts_3d[0][0], box3d_pts_3d[1][0], box3d_pts_3d[2][0], box3d_pts_3d[3][0],
                            box3d_pts_3d[4][0], box3d_pts_3d[5][0], box3d_pts_3d[6][0], box3d_pts_3d[7][0],
                            box3d_pts_3d[0][1], box3d_pts_3d[1][1], box3d_pts_3d[2][1], box3d_pts_3d[3][1],
                            box3d_pts_3d[4][1], box3d_pts_3d[5][1], box3d_pts_3d[6][1], box3d_pts_3d[7][1],
                            box3d_pts_3d[0][2], box3d_pts_3d[1][2], box3d_pts_3d[2][2], box3d_pts_3d[3][2],
                            box3d_pts_3d[4][2], box3d_pts_3d[5][2], box3d_pts_3d[6][2], box3d_pts_3d[7][2],
                            is_obj_list[i]])
        return results

    def LivoxCallback(self, msg):
        header = std_msgs.msg.Header()
        header.frame_id = msg.layout.dim[0].label
        header.stamp = rospy.Time.from_sec(float(msg.layout.dim[1].label))
        vox = np.zeros(HEIGHT * WIDTH * CHANNELS, dtype=int)
        indexes = [id for id in msg.data]
        np.put(vox, indexes, 1)
        # vox[indexes] = 1
        # vox = lib_cpp.get_voxel(indexes, HEIGHT, WIDTH, CHANNELS)
        vox = np.reshape(vox, (HEIGHT, WIDTH, CHANNELS))
        vox = np.expand_dims(vox, axis=0)
        t0 = time.time()
        result = self.detect(vox)
        t1 = time.time()
        print('det_time(ms)', 1000*(t1-t0))
        print('det_numbers', len(result))
        for ii in range(len(result)):
            result[ii][1:9] = list(np.array(result[ii][1:9]))
            result[ii][9:17] = list(np.array(result[ii][9:17]))
            result[ii][17:25] = list(np.array(result[ii][17:25]))
        boxes = result
        marker_array.markers = []
        marker_array_text.markers = []
        for obid in range(len(boxes)):
            ob = boxes[obid]
            tid = 0
            detect_points_set = []
            for i in range(0, 8):
                detect_points_set.append(Point(ob[i+1], ob[i+9], ob[i+17]))

            marker = Marker()
            marker.header = header

            marker.id = obid*2
            marker.action = Marker.ADD
            marker.type = Marker.LINE_LIST

            marker.lifetime = rospy.Duration(0.1)

            marker.color.r = color_map[ob[0]][0]
            marker.color.g = color_map[ob[0]][1]
            marker.color.b = color_map[ob[0]][2]

            # marker.color.a = 1
            # marker.scale.x = 0.2
            marker.color.a = round(np.floor(ob[25]*100)/100, 3)
            marker.scale.x = 0.1
            marker.points = []

            for line in lines:
                marker.points.append(detect_points_set[line[0]])
                marker.points.append(detect_points_set[line[1]])

            # for transform to itrimsg
            marker.text = ob[0]+':'+str(np.floor(ob[25]*100)/100)
            # 

            marker_array.markers.append(marker)
            marker1 = Marker()
            marker1.header = header
            marker1.ns = "basic_shapes"

            marker1.id = obid*2+1
            marker1.action = Marker.ADD

            marker1.type = Marker.TEXT_VIEW_FACING

            marker1.lifetime = rospy.Duration(0.1)

            marker1.color.r = 1  # cr
            marker1.color.g = 1  # cg
            marker1.color.b = 1  # cb

            marker1.color.a = 1
            marker1.scale.z = 1

            marker1.pose.orientation.w = 1.0
            marker1.pose.position.x = (ob[1]+ob[3])/2
            marker1.pose.position.y = (ob[9]+ob[11])/2
            marker1.pose.position.z = (ob[21]+ob[23])/2+1

            marker1.text = ob[0]+':'+str(np.floor(ob[25]*100)/100)

            marker_array_text.markers.append(marker1)

        self.marker_pub.publish(marker_array)
        self.marker_text_pub.publish(marker_array_text)
        print("Pub box ", header.stamp)


if __name__ == '__main__':
    livox = Detector()
    rospy.spin()

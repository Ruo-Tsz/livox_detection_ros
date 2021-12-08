from scipy.spatial import kdtree
import rospy
import tf
import cv2
from itri_msgs.msg import DetectedObject, DetectedObjectArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Point32, Pose, Quaternion, Vector3, PolygonStamped, Polygon
import numpy as np
import copy
import math
import time
import sklearn
from sklearn.neighbors import KDTree
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2


label_mapping = { \
    "0": "car",
    "1": "bus",
    "2": "truck",
    "3": "pedestrian",
    "4": "bimo"}

class MsgConverter(object):
    def __init__(self):
        self.out_msg = DetectedObjectArray()
        self.pub = rospy.Publisher(
            'livox_detection', DetectedObjectArray, queue_size=1000)
    
    def pub_msg(self):
        self.pub.publish(self.out_msg)
    
    def set_msg_property(self, header):
        self.out_msg.sensor_type = 0
        self.out_msg.objects = []
        self.out_msg.header = header
    
    def get_quaterian(self, corners):
        '''
            output x-axis as long width and q w.r.t x-axis
        '''
        edge_1 = np.sum(np.power(np.subtract(corners[0], corners[1]), 2))
        edge_2 = np.sum(np.power(np.subtract(corners[1], corners[2]), 2))
        x_diff = 1
        y_diff = 0
        if edge_1 > edge_2:
            x_diff = corners[0][0] - corners[1][0]
            y_diff = corners[0][1] - corners[1][1]

        else:
            x_diff = corners[1][0] - corners[2][0]
            y_diff = corners[1][1] - corners[2][1]
        # x_diff = corners[0][0] - corners[1][0]
        # y_diff = corners[0][1] - corners[1][1]
        
        phi = math.atan2(y_diff, x_diff)
        # print(phi)
        R = np.zeros((4,4))     
        R[3, 3] = 1
        R[:3, :3] = self.rotx(phi)
        q = tf.transformations.quaternion_from_matrix(R)
        return q

    def get_convex_hull(self, corners):
        hull = PolygonStamped()
        hull.header = self.out_msg.header
        corners_2d = copy.deepcopy(corners[:, :2])
        # (n, 1, 2)
        contour_pt = cv2.convexHull(np.array(corners_2d, dtype=np.float32))

        # find z_min and z_max
        z_max = corners[4][2]
        z_min = corners[0][2]
        # print('z_min: ', z_min)

        polygon = Polygon()
        for idx in range(len(contour_pt)+1):
            geo_pt = Point32()
            geo_pt.x = contour_pt[idx%len(contour_pt)][0][0]
            geo_pt.y = contour_pt[idx%len(contour_pt)][0][1]
            geo_pt.z = z_max
            polygon.points.append(geo_pt)
        
        for idx in range(len(contour_pt)+1):
            geo_pt = Point32()
            geo_pt.x = contour_pt[idx%len(contour_pt)][0][0]
            geo_pt.y = contour_pt[idx%len(contour_pt)][0][1]
            geo_pt.z = z_min
            polygon.points.append(geo_pt)
        
        hull.polygon = polygon
        return hull

    def get_obj_center(self, corners):
        cen_x = 0
        cen_y = 0
        cen_z = 0
        corner_list = copy.deepcopy(corners).tolist()
        for pt in corner_list:
            cen_x = cen_x + pt[0]
            cen_y = cen_y + pt[1]
            cen_z = cen_z + pt[2]    

        cen_x = cen_x / len(corner_list)
        cen_y = cen_y / len(corner_list)
        cen_z = cen_z / len(corner_list)
        
        return Point(cen_x, cen_y, cen_z) 

    def get_obj_dim(self, corners):
        '''
            l: extent in x
            w: extent in y
            h: extent in z
        '''
        cor = copy.deepcopy(corners)

        edge_1 = np.sum(np.power(np.subtract(cor[0], cor[1]), 2))
        edge_2 = np.sum(np.power(np.subtract(cor[1], cor[2]), 2))
        l = 0
        w = 0
        if edge_1 > edge_2:
            l = math.sqrt(edge_1)
            w = math.sqrt(edge_2) 
        else:
            l = math.sqrt(edge_2)
            w = math.sqrt(edge_1) 
        
        h = math.sqrt(np.sum(np.power(np.subtract(cor[0], cor[4]), 2))) 

        # print(l, w, h)

        return Vector3(l, w, h)

    def rotx(self, t):
        ''' 
            t is rad rotate about x axis
        '''
        c = np.cos(t)
        s = np.sin(t)

        return np.array([[c,  -s,  0],
                         [s,  c,  0],
                         [0,  0,  1]])

    def get_cloud(self, tree, cloud, corners_array, center, dim):
        header = self.out_msg.header

        largest_dim = np.max(np.array([dim.x, dim.y, dim.z]))
        z_max = corners_array[-1][-1]
        z_min = corners_array[0][-1]
        cen = np.reshape([center.x, center.y, center.z], (1, -1))
        
        t1 = time.time()
        idx = tree.query_radius(cen, r=largest_dim/2)
        t2 = time.time()
        # print("kd search: ", (t2-t1)*1000)
        pt_idx = idx[0].tolist()
        # print(len(idx[0]), ' pt in cloud')
        # print('largest dim', largest_dim/2)
        # print('hight ', z_max)
        # print('lower ', z_min)

        # filter z pt
        pt_z_idx = []
        for pt, idx in zip(cloud[pt_idx], pt_idx):
            if pt[2] < z_min or pt[2] > z_max:
                continue
            else:
                pt_z_idx.append(idx)
        # print('filter h idx: ', len(pt_z_idx), ' pt')

        t3 = time.time()
        # filter 2d polygon
        pt_2d = cloud[pt_z_idx, 0:2]
        pt_2d = pt_2d.tolist()
        contour = corners_array[:4, :2]
        contour = np.vstack((corners_array[:4, :2], corners_array[0, :2]))
        pt_poly_idx = []
        for pt, idx in zip(pt_2d, pt_z_idx):
            in_poly = cv2.pointPolygonTest(np.array(contour, dtype=np.float32) , (pt[0], pt[1]), False)
            if in_poly != -1:
                pt_poly_idx.append(idx)

        t4 = time.time()
        # print('filter poly: ', (t4-t3)*1000)
        # print('final idx: ', len(pt_poly_idx), ' pt')
        # print(cloud[pt_poly_idx, 0:3])

        pointcloud_msg = pcl2.create_cloud_xyz32(header, cloud[pt_poly_idx, 0:3])
        return pointcloud_msg

    # def convert_msg(self, obj_cls_list, is_obj_list, box3d_pts_3d_list):
    def convert_msg(self, result, cloud):
        ''' 
          elment is list: 
          is_obj_list: score of each obj
          box3d_pts_3d_list: 8*3 array of 8 corners, start from BOTTOM
          result: list of n boxes
        '''   
        # msg = DetectedObjectArray()
        # det_num = len(obj_cls_list)
        # for idx in range(det_num):
        #     corners = copy.deepcopy(box3d_pts_3d_list[idx])

        #     obj = DetectedObject()
        #     obj.header = self.out_msg.header
        #     obj.pose.position = self.get_obj_center(corners)
        #     q = self.get_quaterian(corners)
        #     obj.pose.orientation.x = q[0]
        #     obj.pose.orientation.y = q[1]
        #     obj.pose.orientation.z = q[2]
        #     obj.pose.orientation.w = q[3]
        #     obj.dimensions = self.get_obj_dim(corners)
        #     obj.label = label_mapping[str(int(obj_cls_list[idx]))]
        #     obj.score = np.floor(is_obj_list[idx]*100)/100
        #     obj.convex_hull = self.get_convex_hull(corners)
        #     obj.id = idx

        #     msg.objects.append(obj)

        # self.out_msg.objects = msg.objects
        msg = DetectedObjectArray()
        det_num = len(result)
        tt = time.time()
        tree = KDTree(cloud, leaf_size=5000)
        ttt = time.time()
        cloud_time = 0
        for idx in range(det_num):
            corners = copy.deepcopy(result[idx][1:-1])
            corners_array = np.reshape(np.array(corners), (8, 3), order='F')

            obj = DetectedObject()
            obj.header = self.out_msg.header
            obj.pose.position = self.get_obj_center(corners_array)
            q = self.get_quaterian(corners_array)
            obj.pose.orientation.x = q[0]
            obj.pose.orientation.y = q[1]
            obj.pose.orientation.z = q[2]
            obj.pose.orientation.w = q[3]
            obj.dimensions = self.get_obj_dim(corners_array)
            obj.label = result[idx][0]
            obj.score = np.floor(result[idx][-1]*100)/100
            obj.convex_hull = self.get_convex_hull(corners_array)
            obj.id = idx
            t1 = time.time()
            obj.pointcloud = self.get_cloud(tree, cloud, corners_array, obj.pose.position, obj.dimensions)
            t2 = time.time()
            cloud_time = cloud_time + (t2-t1)*1000
            # print('cloud time: ', (t2-t1)*1000)
            # print('-'*20)
            msg.objects.append(obj)
        
        # print('TOTAL cloud time: ', cloud_time)
        # print('set tree: ', (ttt-tt)*1000)

        self.out_msg.objects = msg.objects


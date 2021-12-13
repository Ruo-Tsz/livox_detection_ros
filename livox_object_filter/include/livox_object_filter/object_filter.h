#ifndef OBJECT_FILTER_H
#define OBJECT_FILTER_H

#include <iostream>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <ros/ros.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <string>
#include <sensor_msgs/point_cloud_conversion.h>

#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <itri_msgs/DetectedObjectArray.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <vector>
#include <std_msgs/Header.h>
#include <tf/transform_datatypes.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace ObjectFilter
{
    class ObjectFilter
    {
        private:
            ros::NodeHandle mNodeHandler;
            ros::Subscriber mSubLidarCloud;
            ros::Subscriber mSubDetectedPoints;
            ros::Publisher mPubFilteredDetectedObjs;
            ros::Publisher mPubPolygonHulls;
            std::vector<sensor_msgs::PointCloud2> mCloudQueue;
            std_msgs::Header mHeader;
            tf::TransformListener mListener;
            tf::StampedTransform mTransform;

            void LidarCallback(
                const sensor_msgs::PointCloud2ConstPtr & msgCloudPtr);
            void DetectedPointsCallback(
                const visualization_msgs::MarkerArrayConstPtr & msgDetectedObjPtr);
            void ObjToItriObj(
                const visualization_msgs::MarkerArray & inObject,
                const sensor_msgs::PointCloud2 & inCloud,
                itri_msgs::DetectedObjectArray & outMsg);
            void GetCorners(
                const std::vector<geometry_msgs::Point> & inCorners,
                std::vector<geometry_msgs::Point> & outCorners);
            geometry_msgs::Point GetObjCenter(const std::vector<geometry_msgs::Point> & inCorners);
            geometry_msgs::Vector3 GetObjDim(const std::vector<geometry_msgs::Point> & inCorners);
            geometry_msgs::Quaternion GetObjOrientation(const std::vector<geometry_msgs::Point> & inCorners);
            geometry_msgs::PolygonStamped GetObjHull(const std::vector<geometry_msgs::Point> & inCorners);
            sensor_msgs::PointCloud2 GetCloud(
                const std::vector<geometry_msgs::Point> & inCorners,
                const pcl::PointCloud<pcl::PointXYZI>::Ptr & inCloudPtr,
                const geometry_msgs::PolygonStamped & inHull);
            float EuclideanDist(
                const geometry_msgs::Point & p1,
                const geometry_msgs::Point & p2);
            void SearchCurrentCloud(
                const std_msgs::Header & inHeader,
                sensor_msgs::PointCloud2 & outCloud);
            bool ptTraversed(
                const geometry_msgs::Point & pt,
                const std::vector<geometry_msgs::Point> & traversedPt);
            void FilteredROI(
                const std::vector<geometry_msgs::Point> & inCorners,
                const pcl::PointCloud<pcl::PointXYZI>::Ptr & inCloud,
                pcl::PointCloud<pcl::PointXYZI>::Ptr & outCloud);

        public:
            ObjectFilter();

    };
}
#endif

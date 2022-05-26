#ifndef MOTION_COMPENSTAION
#define MOTION_COMPENSTAION

#include <iostream>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud2.h>
#include <chrono>
#include <point_container.h>

class MotionCompensation
{
    private:
        ros::NodeHandle mNodeHandler, mPrivateHandler;
        ros::Subscriber mSubScan;
        ros::Publisher mPubScan, mPubScanOrder, mPubGridSlice;
        tf::StampedTransform mLastTransform, mCurrentTransform;
        tf::TransformListener mListener;
        std::string mInputCloud, mOutputCloud;
        std_msgs::Header mLastHeader, mCurrentHeader;
        float mAngResol;
        float mStartAzi;
        bool mClockwise;

        void Callback(const sensor_msgs::PointCloud2ConstPtr &msg);
        void getScanRotation(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr& inCloud,
            float& start_azi,
            bool& clockwise);
        GridSlice buildScanGrid(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr& inCloud,
            const int& col_num);
        pcl::PointCloud<pcl::PointXYZI>::Ptr motionCompensate(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr& inCloud,
            const GridSlice& gridMap,
            const tf::Transform& current_T_last);
        void pubMap(
            const GridSlice& inMapIdx,
            const ros::Publisher& inPublisher,
            const std::string& inFrame_id);

    public:
        MotionCompensation();
        void Run();
};







#endif
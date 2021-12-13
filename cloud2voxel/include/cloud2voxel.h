#ifndef CLOUD_2_VOXEL_H
#define CLOUD_2_VOXEL_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>
#include <std_msgs/Int32MultiArray.h>


namespace Cloud2Voxel
{
    class Cloud2Voxel
    {
    public:
        Cloud2Voxel();
        void CallbackCloud(
            const sensor_msgs::PointCloud2ConstPtr & sensorCloud);
        void Run();

    private:
        ros::NodeHandle nodeHandle;
        ros::Subscriber mSubCloud;
        ros::Publisher mPubVoxel;

        float mMaxX;
        float mMinX;
        float mMaxY;
        float mMinY;
        float mMaxZ;
        float mMinZ;
        float mVoxel;
        float mOverlap;
    };
}

#endif

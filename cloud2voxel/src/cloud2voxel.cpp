#include <cloud2voxel.h>

namespace Cloud2Voxel
{
    Cloud2Voxel::Cloud2Voxel()
    :nodeHandle("~")
    {
        nodeHandle.param("max_x", mMaxX, 89.6f);
        ROS_INFO("max_x[m]: %f", mMaxX);

        nodeHandle.param("min_x", mMinX, -89.6f);
        ROS_INFO("min_x[m]: %f", mMinX);

        nodeHandle.param("max_y", mMaxY, 49.4f);
        ROS_INFO("max_y[m]: %f", mMaxY);

        nodeHandle.param("min_y", mMinY, -49.4f);
        ROS_INFO("min_y[m]: %f", mMinY);

        nodeHandle.param("max_z", mMaxZ, 3.0f);
        ROS_INFO("max_z[m]: %f", mMaxZ);

        nodeHandle.param("min_z", mMinZ, -3.0f);
        ROS_INFO("min_z[m]: %f", mMinZ);

        nodeHandle.param("voxel_size", mVoxel, 0.2f);
        ROS_INFO("voxel_size[m]: %f", mVoxel);

        nodeHandle.param("overlap", mOverlap, 11.2f);
        ROS_INFO("overlap[m]: %f", mOverlap);

        mSubCloud = nodeHandle.subscribe(
            "/velodyne_points", 1, &Cloud2Voxel::CallbackCloud, this);
        mPubVoxel = nodeHandle.advertise<std_msgs::Int32MultiArray>("/voxel", 1);
    }

    void Cloud2Voxel::CallbackCloud(
        const sensor_msgs::PointCloud2ConstPtr & sensorCloud)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*sensorCloud, *pclCloud);

        int H = std::ceil(2.0f * (mMaxX + mOverlap) / mVoxel);
        int W = std::ceil(2.0f * mMaxY / mVoxel);
        int C = std::ceil(2.0f * mMaxZ / mVoxel);

        std_msgs::Int32MultiArray array;

        for (size_t i = 0; i < pclCloud->points.size(); i++)
        {
            float x = pclCloud->points[i].x;
            float y = pclCloud->points[i].y;
            float z = pclCloud->points[i].z;

            if (x > mMinX && x < mMaxX && y > mMinY && y < mMaxY && z > mMinZ && z < mMaxZ)
            {
                // close to ego
                if( std::fabs(x) < 3 || std::fabs(y) < 1.5)
                    continue;

                float tmpChannel = (-z + mMaxZ) / mVoxel;
                int channel = (tmpChannel >=0) ? std::floor(tmpChannel):std::ceil(tmpChannel);

                if (x > -mOverlap)
                {
                    float tmpX = (x - mMinX + 2.0 * mOverlap) / mVoxel;
                    float tmpY = (-y + mMaxY) / mVoxel;
                    int pixelX = (tmpX >=0) ? std::floor(tmpX):std::ceil(tmpX);
                    int pixelY = (tmpY >=0) ? std::floor(tmpY):std::ceil(tmpY);
                    array.data.push_back(pixelX*W*C + pixelY*C + channel);
                }

                if (x < mOverlap)
                {
                    float tmpX = (-x + mOverlap) / mVoxel;
                    float tmpY = (y + mMaxY) / mVoxel;
                    int pixelX = (tmpX >=0) ? std::floor(tmpX):std::ceil(tmpX);
                    int pixelY = (tmpY >=0) ? std::floor(tmpY):std::ceil(tmpY);
                    array.data.push_back(pixelX*W*C + pixelY*C + channel);
                }
            }
        }
        std::sort(array.data.begin(), array.data.end());

        std_msgs::MultiArrayDimension dim1;
        dim1.label = sensorCloud->header.frame_id;
        array.layout.dim.push_back(dim1);

        double timestamp = sensorCloud->header.stamp.toSec();
        std_msgs::MultiArrayDimension dim2;
        dim2.label = std::to_string(timestamp);
        array.layout.dim.push_back(dim2);

        mPubVoxel.publish(array);
    }

    void Cloud2Voxel::Run()
    {
        ros::spin();
    }
}

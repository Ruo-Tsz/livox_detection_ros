/*
 *  The package is used for ego-motion compesation of pointcloud data.
 *  Input:
 *    Global tf: /map for example
 *    Pointcloud with no individual pt timestamp
 */

#include "motion_compensation/motion_compensation.h"


MotionCompensation::MotionCompensation()
: mPrivateHandler("~")
, mLastTransform ()
, mCurrentTransform ()
, mLastHeader ()
, mCurrentHeader ()
, mStartAzi (0)
, mClockwise (false)
{
    mPrivateHandler.param<std::string>("input_cloud", mInputCloud, "velodyne_points");
    mPrivateHandler.param<std::string>("output_cloud", mOutputCloud, "compensated_velodyne_points");
    ROS_INFO("Input cloud: %s", mInputCloud.c_str());
    mPrivateHandler.param<float>("ang_resolusion", mAngResol, 5);
    ROS_INFO("Ang_resolusion: %lf [deg]", mAngResol);
    // mOutputCloud = "Compensated/" + mInputCloud;

    mSubScan = mNodeHandler.subscribe(mInputCloud, 1, &MotionCompensation::Callback, this);
    mPubScan = mNodeHandler.advertise<sensor_msgs::PointCloud2>(mOutputCloud, 1);
    mPubScanOrder = mPrivateHandler.advertise<sensor_msgs::PointCloud2>("scaning_order", 1);
    mPubGridSlice = mPrivateHandler.advertise<visualization_msgs::MarkerArray>("grid_slice_map", 1);
    mBagClient = mNodeHandler.serviceClient<std_srvs::SetBool>("/playback/pause_playback");
}

void MotionCompensation::Run()
{
    ros::Rate loopRate(100);
    while (ros::ok())
    {
        ros::spin();
        loopRate.sleep();
    }
}

void MotionCompensation::PauseBagSrv()
{
    // need to put after listen to tf, before callback return
    std_srvs::SetBool pause_bag_srv;
    pause_bag_srv.request.data = true;
    if(mBagClient.call(pause_bag_srv))
    {
        ROS_INFO("%s for %lf",pause_bag_srv.response.message.c_str(), mCurrentHeader.stamp.toSec());
    }
    else
    {
        ROS_ERROR("Failed to pause bag");
    }    
}

void MotionCompensation::Callback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mLastHeader = mCurrentHeader;
    mLastTransform = mCurrentTransform;
    mCurrentHeader = msg->header;
    std::cout << "Last cloud: " << mLastHeader.stamp << std::endl;
    std::cout << "Get cloud: " << mCurrentHeader.stamp << std::endl;

    try
    {
        mListener.waitForTransform(
            "map", mCurrentHeader.frame_id, mCurrentHeader.stamp, ros::Duration(1.0));
        mListener.lookupTransform(
            "map", mCurrentHeader.frame_id, mCurrentHeader.stamp, mCurrentTransform);
    }
    catch (tf::TransformException ex)
    {
        // Not get current tf
        ROS_WARN("%s",ex.what());
        PauseBagSrv();
        mPubScan.publish(msg);
        return;
    }

    if(mLastHeader.stamp.toSec() == 0 || mLastTransform.stamp_.toSec() == 0)
    {
        // Not get last tf
        std::cout << "First frame, no compensation\n";
        PauseBagSrv();
        mPubScan.publish(msg);
        return;
    }
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr inCloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr outCloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg, *inCloud);

    if(inCloud->points.size() == 0)
    {
        ROS_WARN("No points in scan");
        PauseBagSrv();
        mPubScan.publish(msg);
        return;
    }

    PauseBagSrv();
    getScanRotation(
        inCloud,
        mStartAzi,
        mClockwise);

    int col_num = static_cast<int> (360.0/mAngResol);
    auto gridMap = buildScanGrid(inCloud, col_num);

    // last pose relative to current origin
    tf::Transform current_T_last;
    current_T_last = mCurrentTransform.inverse() * mLastTransform;
    auto _start = std::chrono::system_clock::now();
    outCloud = motionCompensate(inCloud, gridMap, current_T_last);
    auto _end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(_end-_start).count();
    ROS_INFO("motionCompensate : %f ms", elapsed);

    sensor_msgs::PointCloud2 outCloud_msg;
    pcl::toROSMsg(*outCloud, outCloud_msg);
    outCloud_msg.header = msg->header;
    mPubScan.publish(outCloud_msg);
    std::cout << "Pub cloud\n\n";

    pubMap(gridMap, mPubGridSlice, mCurrentHeader.frame_id);
}

void MotionCompensation::getScanRotation(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& inCloud,
    float& start_azi,
    bool& clockwise)
{
    /*
        Get starting scan azimuth in pointcloud coordinate and rotate orientation.
        This function is for non-stamped pointcloud.
        @Param
            INPUT:
                inCloud: original input cloud
            OUTPUT:
                start_azi: record first pt azimuth (degree) in poincloud sequence
                clockwise: scanning orientation
    */
    pcl::PointCloud<pcl::PointXYZI>::Ptr outCloud(new pcl::PointCloud<pcl::PointXYZI>);
    int counter = 0;
    for(auto pt: inCloud->points)
    {
        pcl::PointXYZI pt_out;
        pt_out = pt;
        pt_out.intensity = counter;
        outCloud->points.push_back(pt_out);
        counter++;
    }
    sensor_msgs::PointCloud2 outCloud_msg;
    pcl::toROSMsg(*outCloud, outCloud_msg);
    outCloud_msg.header = mCurrentHeader;
    mPubScanOrder.publish(outCloud_msg);

    // check scanning direction by points' order in cloud
    float first_azimuth=0, second_azimuth=0, first_counter=0, second_counter=0;
    float window_size = 10;
    for(int i = 0; i < inCloud->points.size(); i++)
    {
        float azi = std::atan2(inCloud->points[i].y, inCloud->points[i].x) * 180 / M_PI;
        azi = (azi < 0) ? azi + 360 : azi;
        if(i == 0)
            start_azi = azi;

        if(i < inCloud->points.size()/window_size)
        {
            first_azimuth += azi;
            first_counter++;
        }
        else if (i < inCloud->points.size()/(window_size/2))
        {
            second_azimuth += azi; 
            second_counter++;
        }
        else
            break;
    }
    first_azimuth /= first_counter;
    second_azimuth /= second_counter;

    if(second_azimuth < first_azimuth)
        clockwise = true;
    else
        clockwise = false;

    std::cout << "start:" << start_azi << "; first: " << first_azimuth << "; second: " << second_azimuth << "; clockwise: " << clockwise << std::endl;
}

GridSlice MotionCompensation::buildScanGrid(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& inCloud,
    const int& col_num)
{
    /*
        Build 1-d grid map in angular direction.
        Base on coordinate of poincloud, NOT from SCANNING ORDER.
        Map store pt index in original inCloud.
        @Param
            INPUT:
                inCloud: pointcloud
                col_num: total num of grid
            OUTPUT:
                1-d map of grid slices
            
    */
    GridSlice polarGrid(col_num);
    
    int col_idx;
    float range, azimuth;
    int pt_idx = 0;
    for(const auto& pt: inCloud->points)
    {
        azimuth = std::atan2(pt.y, pt.x)*180/M_PI;
        if(azimuth < 0) azimuth+=360;

        col_idx = (floor)(azimuth/mAngResol);

        if(col_idx > col_num-1) 
        {
            // if(row_idx > row_num-1)
            //     std::cout << "Out of range: " << range << std::endl;
            // else
            //     std::cout << "\033[1;33mError azi\033[0m, col_idx: " << col_idx << ", azimuth: " << azimuth << std::endl;
            pt_idx++;
            continue;
        }

        polarGrid[col_idx].points().push_back(pt_idx);
        pt_idx++;
    }

    return polarGrid;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr MotionCompensation::motionCompensate(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& inCloud,
    const GridSlice& gridMap,
    const tf::Transform& current_T_last)
{
    /*
        Ego-motion compensation by transformation of 2 scans.
        Compensate each grids' point in batch by revolution ratio of entire scan. 
        @Param:
            INPUT:
                inCloud
                gridMap: built gridMap
                current_T_last: transformation of last scan ego-frame respect to current scan ego-frame
            OUTPUT:
                compensated pointcloud
    */
    pcl::PointCloud<pcl::PointXYZI>::Ptr outCloud(new pcl::PointCloud<pcl::PointXYZI>);
    outCloud->header = inCloud->header;

    std_msgs::Header gridStamp;
    gridStamp.frame_id = inCloud->header.frame_id;
    int basetime = mLastHeader.stamp.toNSec();
    int dt = mCurrentHeader.stamp.toNSec() - basetime;
    std::cout << "dt: " << dt << std::endl;
    int timestep = 0;
    float rev = 0;

    tf::StampedTransform gridTransform;

    int start_index = (floor)(mStartAzi/mAngResol);
    std::cout << "start_index: " << start_index << "; mStartAzi: " << mStartAzi << std::endl;
    if(mClockwise)
    {
        for(GridSlice::const_iterator it = gridMap.begin()+start_index; it != gridMap.begin()-1; it--)
        {
            int n_traverse_grid = start_index - (it - gridMap.begin());
            timestep = (n_traverse_grid + 0.5)* mAngResol / 360 * dt;
            rev = (n_traverse_grid + 0.5)* mAngResol / 360;
            gridStamp.stamp.sec = int((basetime + timestep)/1e9);
            gridStamp.stamp.nsec = int(basetime + timestep) % int(1e9);

            // std::cout << "n_traverse_grid: " << n_traverse_grid << std::endl;
            // std::cout << "timestep: " << timestep << std::endl;
            // std::cout << "rev: " << rev << std::endl;

            // k-1's(grid) pose relative to k origin
            tf::Transform current_T_grid;
            
            // interpolate transform by revolusion
            current_T_grid.setOrigin(tf::Vector3(
                current_T_last.getOrigin().x()*(1-rev),
                current_T_last.getOrigin().y()*(1-rev),
                current_T_last.getOrigin().z()*(1-rev)));
            current_T_grid.setRotation(tf::Quaternion(
                current_T_last.getRotation().x()*(1-rev),
                current_T_last.getRotation().y()*(1-rev),
                current_T_last.getRotation().z()*(1-rev),
                current_T_last.getRotation().w()*(1-rev)));
            
            for(const auto& pt_idx: it->points())
            {
                tf::Point pt_ori(inCloud->points[pt_idx].x, inCloud->points[pt_idx].y, inCloud->points[pt_idx].z);
                tf::Point pt_comp = current_T_grid * pt_ori;
                pcl::PointXYZI pt_final;
                pt_final.x = pt_comp.x();
                pt_final.y = pt_comp.y();
                pt_final.z = pt_comp.z();
                pt_final.intensity = inCloud->points[pt_idx].intensity;
                outCloud->points.push_back(pt_final);
            }
        }
        // for(GridSlice::const_reverse_iterator it = gridMap.rbegin(); it != gridMap.begin()+start_index; it++)
        for(GridSlice::const_iterator it = gridMap.end()-1; it != gridMap.begin()+start_index; it--)
        {
            int n_traverse_grid = (start_index+1) + (gridMap.end()-1 - it);
            timestep = (n_traverse_grid + 0.5)* mAngResol / 360 * dt;
            rev = (n_traverse_grid + 0.5)* mAngResol / 360;
            gridStamp.stamp.sec = int((basetime + timestep)/1e9);
            gridStamp.stamp.nsec = int(basetime + timestep) % int(1e9);

            // std::cout << "n_traverse_grid: " << n_traverse_grid << std::endl;
            // std::cout << "timestep: " << timestep << std::endl;
            // std::cout << "rev: " << rev << std::endl;

            // k-1's(grid) pose relative to k origin
            tf::Transform current_T_grid;
            
            // interpolate transform by revolusion
            current_T_grid.setOrigin(tf::Vector3(
                current_T_last.getOrigin().x()*(1-rev),
                current_T_last.getOrigin().y()*(1-rev),
                current_T_last.getOrigin().z()*(1-rev)));
            current_T_grid.setRotation(tf::Quaternion(
                current_T_last.getRotation().x()*(1-rev),
                current_T_last.getRotation().y()*(1-rev),
                current_T_last.getRotation().z()*(1-rev),
                current_T_last.getRotation().w()*(1-rev)));
            
            for(const auto& pt_idx: it->points())
            {
                tf::Point pt_ori(inCloud->points[pt_idx].x, inCloud->points[pt_idx].y, inCloud->points[pt_idx].z);
                tf::Point pt_comp = current_T_grid * pt_ori;
                pcl::PointXYZI pt_final;
                pt_final.x = pt_comp.x();
                pt_final.y = pt_comp.y();
                pt_final.z = pt_comp.z();
                pt_final.intensity = inCloud->points[pt_idx].intensity;
                outCloud->points.push_back(pt_final);
            }
        }
    }
    else
    {
        for(GridSlice::const_iterator it = gridMap.begin()+start_index; it != gridMap.end(); it++)
        {
            int n_traverse_grid = it - gridMap.begin() - start_index;
            timestep = (n_traverse_grid + 0.5)* mAngResol / 360 * dt;
            rev = (n_traverse_grid + 0.5)* mAngResol / 360;
            gridStamp.stamp.sec = int((basetime + timestep)/1e9);
            gridStamp.stamp.nsec = int(basetime + timestep) % int(1e9);

            // k-1's(grid) pose relative to k origin
            tf::Transform current_T_grid;
            
            // interpolate transform by revolusion
            current_T_grid.setOrigin(tf::Vector3(
                current_T_last.getOrigin().x()*(1-rev),
                current_T_last.getOrigin().y()*(1-rev),
                current_T_last.getOrigin().z()*(1-rev)));
            current_T_grid.setRotation(tf::Quaternion(
                current_T_last.getRotation().x()*(1-rev),
                current_T_last.getRotation().y()*(1-rev),
                current_T_last.getRotation().z()*(1-rev),
                current_T_last.getRotation().w()*(1-rev)));
            
            for(const auto& pt_idx: it->points())
            {
                tf::Point pt_ori(inCloud->points[pt_idx].x, inCloud->points[pt_idx].y, inCloud->points[pt_idx].z);
                tf::Point pt_comp = current_T_grid * pt_ori;
                pcl::PointXYZI pt_final;
                pt_final.x = pt_comp.x();
                pt_final.y = pt_comp.y();
                pt_final.z = pt_comp.z();
                pt_final.intensity = inCloud->points[pt_idx].intensity;
                outCloud->points.push_back(pt_final);
            }
        }
        for(GridSlice::const_iterator it = gridMap.begin(); it != gridMap.begin()+start_index; it++)
        {
            int n_traverse_grid = (gridMap.size() - start_index) + (it - gridMap.begin());
            timestep = (n_traverse_grid + 0.5)* mAngResol / 360 * dt;
            rev = (n_traverse_grid + 0.5)* mAngResol / 360;
            gridStamp.stamp.sec = int((basetime + timestep)/1e9);
            gridStamp.stamp.nsec = int(basetime + timestep) % int(1e9);

            // k-1's(grid) pose relative to k origin
            tf::Transform current_T_grid;
            
            // interpolate transform by revolusion
            current_T_grid.setOrigin(tf::Vector3(
                current_T_last.getOrigin().x()*(1-rev),
                current_T_last.getOrigin().y()*(1-rev),
                current_T_last.getOrigin().z()*(1-rev)));
            current_T_grid.setRotation(tf::Quaternion(
                current_T_last.getRotation().x()*(1-rev),
                current_T_last.getRotation().y()*(1-rev),
                current_T_last.getRotation().z()*(1-rev),
                current_T_last.getRotation().w()*(1-rev)));
            
            for(const auto& pt_idx: it->points())
            {
                tf::Point pt_ori(inCloud->points[pt_idx].x, inCloud->points[pt_idx].y, inCloud->points[pt_idx].z);
                tf::Point pt_comp = current_T_grid * pt_ori;
                pcl::PointXYZI pt_final;
                pt_final.x = pt_comp.x();
                pt_final.y = pt_comp.y();
                pt_final.z = pt_comp.z();
                pt_final.intensity = inCloud->points[pt_idx].intensity;
                outCloud->points.push_back(pt_final);
            }
        }
    }
    return outCloud;
}

void MotionCompensation::pubMap(
    const GridSlice& inMapIdx,
    const ros::Publisher& inPublisher,
    const std::string& inFrame_id)
{
    visualization_msgs::MarkerArray slice_grids;

    // pub grid line, iterate col(angular slice) first
    for(int j = 0; j < inMapIdx.size(); j++)
    {
        visualization_msgs::Marker straight_line_marker;
        straight_line_marker.type = visualization_msgs::Marker::LINE_STRIP;
        straight_line_marker.header.stamp = ros::Time::now();
        straight_line_marker.header.frame_id = inFrame_id;
        straight_line_marker.ns = "straight_lines";
        straight_line_marker.id = j;
        straight_line_marker.action = visualization_msgs::Marker::ADD;
        straight_line_marker.scale.x = 0.05;
        straight_line_marker.color.r = 242.0/255.0;
        straight_line_marker.color.g = 92.0/255.0;
        straight_line_marker.color.b = 192.0/255.0;
        straight_line_marker.color.a = 0.7;

        for(int i = 0; i < 70; i+=10)
        {
            float azi = j*mAngResol;
            geometry_msgs::Point pt;
            pt.x = i * std::cos(azi*M_PI/180);
            pt.y = i * std::sin(azi*M_PI/180);
            pt.z = -1.9;
            straight_line_marker.points.push_back(pt);
        }
        slice_grids.markers.push_back(straight_line_marker);
    }

    visualization_msgs::Marker scan_start_line_marker;
    scan_start_line_marker.type = visualization_msgs::Marker::LINE_STRIP;
    scan_start_line_marker.header.stamp = ros::Time::now();
    scan_start_line_marker.header.frame_id = inFrame_id;
    scan_start_line_marker.ns = "scan_start_lines";
    scan_start_line_marker.id = 0;
    scan_start_line_marker.action = visualization_msgs::Marker::ADD;
    scan_start_line_marker.scale.x = 0.1;
    scan_start_line_marker.color.r = 0;
    scan_start_line_marker.color.g = 1;
    scan_start_line_marker.color.b = 0;
    scan_start_line_marker.color.a = 0.7;
    geometry_msgs::Point pt;
    pt.x = 0;
    pt.y = 0;
    pt.z = -1.9;
    scan_start_line_marker.points.push_back(pt);
    pt.x = 60 * std::cos(mStartAzi*M_PI/180);
    pt.y = 60 * std::sin(mStartAzi*M_PI/180);
    pt.z = -1.9;
    scan_start_line_marker.points.push_back(pt);
    slice_grids.markers.push_back(scan_start_line_marker);
    
    inPublisher.publish(slice_grids);
}

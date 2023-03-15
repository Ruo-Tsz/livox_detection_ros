#include <livox_object_filter/object_filter.h>
#include <pcl/console/time.h>
#include <chrono>
#include <pcl/kdtree/kdtree_flann.h>


static std::chrono::system_clock::time_point _start, _end;
namespace ObjectFilter
{
    static std::string INPUT_CLOUD_TOPIC;
    static std::string INPUT_OBJECT_TOPIC;
    static std::string OUTPUT_TOPIC;
    static std::string OUTPUT_HULL_TOPIC;

    ObjectFilter::ObjectFilter()
        :mNodeHandler("~")
        ,mCloudQueue({})
    {
        mNodeHandler.param<std::string>("input_cloud_topic", INPUT_CLOUD_TOPIC, "/velodyne");
        ROS_INFO("input_cloud_topic: %s",INPUT_CLOUD_TOPIC.c_str());
        mNodeHandler.param<std::string>("input_object_topic", INPUT_OBJECT_TOPIC, "/detect_box3d");
        ROS_INFO("input_object_topic: %s",INPUT_OBJECT_TOPIC.c_str());
        mNodeHandler.param<std::string>("output_topic", OUTPUT_TOPIC, "/filtered_lidar_objects");
        ROS_INFO("output_topic: %s",OUTPUT_TOPIC.c_str());
        mNodeHandler.param<std::string>("output_hull", OUTPUT_HULL_TOPIC, "/filtered_cluster_hulls");
        ROS_INFO("output_hull_topic: %s",OUTPUT_HULL_TOPIC.c_str());

        mPubFilteredDetectedObjs =
            mNodeHandler.advertise<itri_msgs::DetectedObjectArray>(
            OUTPUT_TOPIC, 1);
        mPubPolygonHulls =
            mNodeHandler.advertise<jsk_recognition_msgs::PolygonArray>(
            OUTPUT_HULL_TOPIC, 1);
        mSubLidarCloud = mNodeHandler.subscribe(
            INPUT_CLOUD_TOPIC, 1,
            &ObjectFilter::LidarCallback, this);
        mSubDetectedPoints = mNodeHandler.subscribe(
            INPUT_OBJECT_TOPIC, 1,
            &ObjectFilter::DetectedPointsCallback, this);
        
        mTransform.setIdentity();
    }


    void ObjectFilter::LidarCallback(
        const sensor_msgs::PointCloud2ConstPtr & msgCloudPtr)
    {
        std::cout << "\033[1;35mGet Lidar \033[0m" << msgCloudPtr->header << std::endl;
        sensor_msgs::PointCloud2 msg = *msgCloudPtr;
        mCloudQueue.push_back(msg);
    }

    void ObjectFilter::DetectedPointsCallback(
            const visualization_msgs::MarkerArrayConstPtr & msgDetectedObjPtr)
    {
        if(msgDetectedObjPtr->markers.size() == 0)
            return;

        mHeader = msgDetectedObjPtr->markers[0].header;
        
        // pcl::console::TicToc total;
        // total.tic();
        _start = std::chrono::system_clock::now();

        visualization_msgs::MarkerArray obj_msg = *msgDetectedObjPtr;
        sensor_msgs::PointCloud2 current_cloud;
        std::chrono::system_clock::time_point _search_start = std::chrono::system_clock::now();
        SearchCurrentCloud(obj_msg.markers[0].header, current_cloud);
        std::chrono::system_clock::time_point _search_end = std::chrono::system_clock::now();
        double search_time = std::chrono::duration_cast<std::chrono::milliseconds>(_search_end-_search_start).count();

        if(current_cloud.width*current_cloud.height == 0)
        {
            std::cout<<"Not get matched cloud\n";
            return;
        }

        std::cout << "In DetectedPointsCallback\n";
        std::cout << "Get box pts: " << msgDetectedObjPtr->markers[0].header.stamp << std::endl;
        std::cout << "Get cloud pts: " << current_cloud.header << std::endl;

        itri_msgs::DetectedObjectArray out_msg;
        jsk_recognition_msgs::PolygonArray out_msg_polygon;

        //  transform to global frame
        try
        {
            // last 3 digit in nano-sec would shift in markers, use cloud stamp for correct tf
            mListener.waitForTransform("/map", "/base_link", current_cloud.header.stamp, ros::Duration(1.0));
            mListener.lookupTransform("/map", "/base_link", current_cloud.header.stamp, mTransform);
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s", ex.what());
            ros::Duration(1.0).sleep();
        }
        // 

        out_msg.sensor_type = 0;
        out_msg.header = mHeader;
        out_msg_polygon.header = mHeader;
        ObjToItriObj(obj_msg, current_cloud, out_msg);

        out_msg.header.frame_id = "/map";
        out_msg_polygon.header.frame_id = "/map";

        for (size_t i = 0; i < out_msg.objects.size(); i++)
        {
            geometry_msgs::PoseStamped poseOut;

            poseOut.header = out_msg.header;
            poseOut.pose = out_msg.objects[i].pose;
            tf::Transform inObjectsPose;
            inObjectsPose.setOrigin(tf::Vector3(
                out_msg.objects[i].pose.position.x,
                out_msg.objects[i].pose.position.y,
                out_msg.objects[i].pose.position.z));
            inObjectsPose.setRotation(tf::Quaternion(
                out_msg.objects[i].pose.orientation.x,
                out_msg.objects[i].pose.orientation.y,
                out_msg.objects[i].pose.orientation.z,
                out_msg.objects[i].pose.orientation.w));
            tf::poseTFToMsg(mTransform * inObjectsPose, poseOut.pose);

            geometry_msgs::PolygonStamped inObjectsHull;
            for (size_t j = 0; j < out_msg.objects[i].convex_hull.polygon.points.size(); j++)
            {
                inObjectsHull.header = out_msg.header;
                tf::Point pt(
                    out_msg.objects[i].convex_hull.polygon.points[j].x,
                    out_msg.objects[i].convex_hull.polygon.points[j].y,
                    out_msg.objects[i].convex_hull.polygon.points[j].z);
                tf::Point globalpt = mTransform * pt;
                geometry_msgs::Point32 point;
                point.x = globalpt.x();
                point.y = globalpt.y();
                point.z = globalpt.z();
                inObjectsHull.polygon.points.push_back(point);
            }
            out_msg_polygon.polygons.push_back(inObjectsHull);
            out_msg_polygon.labels.push_back(i);

            sensor_msgs::PointCloud2 inCloud = out_msg.objects[i].pointcloud;
            sensor_msgs::PointCloud2Ptr transformCloudPtr(new sensor_msgs::PointCloud2);
            pcl_ros::transformPointCloud("/map", mTransform, inCloud, *transformCloudPtr);

            itri_msgs::DetectedObject globalObj;
            globalObj = out_msg.objects[i];
            globalObj.header = out_msg.header;
            globalObj.pose = poseOut.pose;
            globalObj.convex_hull = inObjectsHull;
            globalObj.pointcloud = *transformCloudPtr;

            out_msg.objects[i] = globalObj;
        }
        // 

        // for(size_t i = 0; i < out_msg.objects.size(); i++)
        // {
        //     out_msg_polygon.polygons.push_back(out_msg.objects[i].convex_hull);
        //     out_msg_polygon.labels.push_back(i);
        // }

        mPubFilteredDetectedObjs.publish(out_msg);
        mPubPolygonHulls.publish(out_msg_polygon);

        // std::cout <<"Total time: " << total.toc()/1000 << " ms\n";

        _end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(_end-_start).count();
        // std::cout << obj_msg.markers.size() << " objs\n";
        std::cout <<"Search cloud time: " << search_time << " ms\n";
        std::cout <<"Total time: " << elapsed << " ms\n";

        std::cout << "Pub " << out_msg.header.stamp << std::endl; 
        std::cout << "--------------------------\n";
    }

    bool ObjectFilter::ptTraversed(
        const geometry_msgs::Point & pt,
        const std::vector<geometry_msgs::Point> & traversedPt)
    {
        for(const auto& checkedPt : traversedPt)
        {
            if(pt.x == checkedPt.x &&
                pt.y == checkedPt.y && 
                pt.z == checkedPt.z)
                return true;
        }
        return false;
    }

    float ObjectFilter::EuclideanDist(
        const geometry_msgs::Point & p1,
        const geometry_msgs::Point & p2)
    {   
        return std::sqrt( std::pow(p1.x-p2.x, 2) + std::pow(p1.y-p2.y, 2) + std::pow(p1.z-p2.z, 2) );
    }

    geometry_msgs::Point ObjectFilter::GetObjCenter(
        const std::vector<geometry_msgs::Point> & inCorners)
    {
        float mean_x = 0;
        float mean_y = 0;
        float mean_z = 0;
        for(const auto& pt : inCorners)
        {
            mean_x += pt.x;
            mean_y += pt.y;
            mean_z += pt.z;
        }
        mean_x /= inCorners.size();
        mean_y /= inCorners.size();
        mean_z /= inCorners.size();
        geometry_msgs::Point cen;
        cen.x = mean_x;
        cen.y = mean_y;
        cen.z = mean_z;
        return cen; 
    }

    geometry_msgs::Quaternion ObjectFilter::GetObjOrientation(
        const std::vector<geometry_msgs::Point> & inCorners)
    {
        geometry_msgs::Quaternion outQ;
        float edge_1 = EuclideanDist(inCorners[0], inCorners[1]);
        float edge_2 = EuclideanDist(inCorners[1], inCorners[2]);
        float x_diff = 1;
        float y_diff = 0;
        if (edge_1 > edge_2)
        {
            x_diff = inCorners[0].x - inCorners[1].x;
            y_diff = inCorners[0].y - inCorners[1].y;
        }
        else
        {
            x_diff = inCorners[1].x - inCorners[2].x;
            y_diff = inCorners[1].y - inCorners[2].y;        
        }
        float phi = std::atan2(y_diff, x_diff);
        tf::quaternionTFToMsg(tf::createQuaternionFromYaw(phi) , outQ);
        return outQ;
    }

    void ObjectFilter::GetCorners(
        const std::vector<geometry_msgs::Point> & inCorners,
        std::vector<geometry_msgs::Point> & outCorners)
    {
        std::vector<geometry_msgs::Point> corners_pt;
        for(const auto& pt : inCorners)
        {
            if(!ptTraversed(pt, corners_pt))
                corners_pt.push_back(pt);
        }
        outCorners = corners_pt;
    }

    geometry_msgs::PolygonStamped ObjectFilter::GetObjHull(
        const std::vector<geometry_msgs::Point> & inCorners)
    {
        // float z_min = std::numeric_limits<float>::max();
        // float z_max = std::numeric_limits<float>::min();
        // std::vector<cv::Point2f> pt_2d;
        // for (size_t i = 0; i < inCorners.size(); i++)
        // {
        //     cv::Point2f pt;
        //     pt.x = inCorners[i].x;
        //     pt.y = inCorners[i].y;
        //     pt_2d.push_back(pt);

        //     if(inCorners[i].z < z_min)    z_min = inCorners[i].z;
        //     if(inCorners[i].z > z_max)    z_max = inCorners[i].z;
        // }

        // std::vector<cv::Point2f> hull_2d;
        // cv::convexHull(pt_2d, hull_2d);
        
        // [0, 1, 2, 3, 0, 4, 5, 6, 7, 4]
        geometry_msgs::PolygonStamped hull;
        // hull.header = mHeader;
        // for (size_t i = 0; i < hull_2d.size()+1; i++)
        // {
        //     geometry_msgs::Point32 pt;
        //     pt.x = hull_2d[i%hull_2d.size()].x;
        //     pt.y = hull_2d[i%hull_2d.size()].y;
        //     pt.z = z_min;
        //     hull.polygon.points.push_back(pt);
        // }

        // for (size_t i = 0; i < hull_2d.size()+1; i++)
        // {
        //     geometry_msgs::Point32 pt;
        //     pt.x = hull_2d[i%hull_2d.size()].x;
        //     pt.y = hull_2d[i%hull_2d.size()].y;
        //     pt.z = z_max;
        //     hull.polygon.points.push_back(pt);
        // }
        hull.header = mHeader;
        hull.polygon.points.resize(inCorners.size() + 2);
        int singleLayerSize = inCorners.size() / 2;
        for (size_t i = 0; i < singleLayerSize + 1; i++)
        {
            geometry_msgs::Point32 pt;
            float x = inCorners[i%singleLayerSize].x;
            float y = inCorners[i%singleLayerSize].y;

            for (size_t j = 0; j < 2; j++)
            {
                hull.polygon.points[i + j * (singleLayerSize + 1)].x = x;
                hull.polygon.points[i + j * (singleLayerSize + 1)].y = y;
                hull.polygon.points[i + j * (singleLayerSize + 1)].z =
                    inCorners[i%singleLayerSize + j * singleLayerSize].z;
            }
        }

        // for (size_t i = inCorners.size()/2; i < inCorners.size(); i++)
        // {
        //     geometry_msgs::Point32 pt;
        //     pt.x = inCorners[i%inCorners.size()].x;
        //     pt.y = inCorners[i%inCorners.size()].y;
        //     pt.z = inCorners[i%inCorners.size()].z;
        //     hull.polygon.points.push_back(pt);
        // }
        // geometry_msgs::Point32 pt;
        // pt.x = inCorners[4].x;
        // pt.y = inCorners[4].y;
        // pt.z = inCorners[4].z;
        // hull.polygon.points.push_back(pt);
        return hull;
    }

    void ObjectFilter::FilteredROI(
        const std::vector<geometry_msgs::Point> & inCorners,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr & inCloud,
        pcl::PointCloud<pcl::PointXYZI>::Ptr & outCloud)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr filteredCloud (new pcl::PointCloud<pcl::PointXYZI>);
        float edge_1 = EuclideanDist(inCorners[0], inCorners[1]);
        float edge_2 = EuclideanDist(inCorners[1], inCorners[2]);
        float edge_3 = EuclideanDist(inCorners[0], inCorners[4]);
        float diameter = (edge_1 > edge_2) ? edge_1 : edge_2;
        diameter = (diameter > edge_3) ? diameter : edge_3;
        // std::cout << diameter << std::endl;

        pcl::KdTreeFLANN<pcl::PointXYZI>  kdtree;
        kdtree.setInputCloud(inCloud);
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        auto cen = GetObjCenter(inCorners);
        pcl::PointXYZI searchPoint;
        searchPoint.x = cen.x;
        searchPoint.y = cen.y;
        searchPoint.z = cen.z;

        if ( kdtree.radiusSearch (searchPoint, diameter/2, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
        {
            for(const auto& idx : pointIdxRadiusSearch)
            {
                filteredCloud->points.push_back(inCloud->points[idx]);
            }
        }
        outCloud = filteredCloud;
    }

    sensor_msgs::PointCloud2 ObjectFilter::GetCloud(
        const std::vector<geometry_msgs::Point> & inCorners,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr & inCloudPtr,
        const geometry_msgs::PolygonStamped & inHull)
    {
        float x_min = std::numeric_limits<float>::max();
        float x_max = std::numeric_limits<float>::min();
        float y_min = std::numeric_limits<float>::max();
        float y_max = std::numeric_limits<float>::min();
        float z_min = std::numeric_limits<float>::max();
        float z_max = std::numeric_limits<float>::min();
        for (size_t i = 0; i < inCorners.size(); i++)
        {
            if(inCorners[i].x < x_min)    x_min = inCorners[i].x;
            if(inCorners[i].x > x_max)    x_max = inCorners[i].x;
            if(inCorners[i].y < y_min)    y_min = inCorners[i].y;
            if(inCorners[i].y > y_max)    y_max = inCorners[i].y;
            if(inCorners[i].z < z_min)    z_min = inCorners[i].z;
            if(inCorners[i].z > z_max)    z_max = inCorners[i].z;
        }
        pcl::PointCloud<pcl::PointXYZI>::Ptr inCloudFilteredPtr (new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr inCloudFilteredPtr_pass (new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr outCloudPtr (new pcl::PointCloud<pcl::PointXYZI>);
        
        std::chrono::system_clock::time_point _pass_start, _pass_end;
        _pass_start = std::chrono::system_clock::now();
        pcl::PassThrough<pcl::PointXYZI> pass; 	
        pass.setInputCloud(inCloudPtr);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(z_min, z_max);
        pass.filter(*inCloudFilteredPtr_pass);  

        pass.setInputCloud(inCloudFilteredPtr_pass);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(x_min, x_max);
        pass.filter(*inCloudFilteredPtr_pass);  

        pass.setInputCloud(inCloudFilteredPtr_pass);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(y_min, y_max);
        pass.filter(*inCloudFilteredPtr_pass);  
        // std::chrono::system_clock::time_point _kd_start, _kd_end;
        // _kd_start = std::chrono::system_clock::now();
        // // FilteredROI(inCorners, inCloudPtr, inCloudFilteredPtr, remainCloud);
        // FilteredROI(inCorners, inCloudFilteredPtr_pass, inCloudFilteredPtr);
        // _kd_end = std::chrono::system_clock::now();
        _pass_end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(_pass_end-_pass_start).count();

        // std::cout << "Pass through: " << inCloudFilteredPtr_pass->points.size() << " pt, " << elapsed <<" us\n";

        std::chrono::system_clock::time_point _clip_start, _clip_end;
        _clip_start = std::chrono::system_clock::now();


        // 1
        // std::vector<cv::Point2f> cv_hulls;
        // for (size_t i = 0 ; i < inHull.polygon.points.size()/2-1; i++)
        // {
        //     cv::Point2f pt;
        //     pt.x = inHull.polygon.points[i].x;
        //     pt.y = inHull.polygon.points[i].y;
        //     cv_hulls.push_back(pt);
        // }

        // for (const auto& pt : inCloudFilteredPtr_pass->points)
        // {
        //     cv::Point2f cv_pt;
        //     cv_pt.x = pt.x;
        //     cv_pt.y = pt.y;

        //     // check if pt is not inside polygon hulls (return value = -1 if outside)
        //     if (cv::pointPolygonTest(cv_hulls, cv_pt, false) != -1)
        //         outCloudPtr->points.push_back(pt);
        // }

        // 2
        float edge_x1 = inHull.polygon.points[0].x - inHull.polygon.points[1].x;
        float edge_y1 = inHull.polygon.points[0].y - inHull.polygon.points[1].y;
        float edge_x2 = inHull.polygon.points[2].x - inHull.polygon.points[1].x;
        float edge_y2 = inHull.polygon.points[2].y - inHull.polygon.points[1].y;
        float dotEdg1 = edge_x1 * edge_x1 + edge_y1 * edge_y1;
        float dotEdg2 = edge_x2 * edge_x2 + edge_y2 * edge_y2;

        for (const auto& pt : inCloudFilteredPtr_pass->points)
        {
            // check if pt is not inside BoundingBox
            float edge_x3 = pt.x - inHull.polygon.points[1].x;
            float edge_y3 = pt.y - inHull.polygon.points[1].y;
            float dotEdg13 = edge_x1 * edge_x3 + edge_y1 * edge_y3;
            float dotEdg23 = edge_x2 * edge_x3 + edge_y2 * edge_y3;

            if (dotEdg13 >= 0 && dotEdg13 <= dotEdg1 &&
                dotEdg23 >=0  && dotEdg23 <= dotEdg2)
                outCloudPtr->points.push_back(pt);
        }


        _clip_end = std::chrono::system_clock::now();
        double clip_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(_clip_end-_clip_start).count();
        // std::cout <<  clip_elapsed <<" us\n\n";


        sensor_msgs::PointCloud2 outCloud;
        pcl::toROSMsg(*outCloudPtr, outCloud);
        outCloud.header = mHeader;
        return outCloud;
    }  

    geometry_msgs::Vector3 ObjectFilter::GetObjDim(
        const std::vector<geometry_msgs::Point> & inCorners)
    {
        float edge_1 = EuclideanDist(inCorners[0], inCorners[1]);
        float edge_2 = EuclideanDist(inCorners[1], inCorners[2]);
        float edge_3 = std::fabs(inCorners[0].z - inCorners[4].z);

        geometry_msgs::Vector3 dim;
        dim.x = (edge_1 > edge_2) ? edge_1 : edge_2;
        dim.y = (edge_1 < edge_2) ? edge_1 : edge_2;
        dim.z = edge_3;
        return dim;
    }

    void ObjectFilter::ObjToItriObj(
        const visualization_msgs::MarkerArray & inObject,
        const sensor_msgs::PointCloud2 & inCloud,
        itri_msgs::DetectedObjectArray & outMsg)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr inCloudPtr (new pcl::PointCloud<pcl::PointXYZI>);   
        pcl::fromROSMsg(inCloud, *inCloudPtr);
        
        float  total_cloud_time = 0;
        float  total_convex_time = 0;
        int objs_size = inObject.markers.size();
        for(size_t i = 0; i < objs_size; i++)
        {
            itri_msgs::DetectedObject obj;
            obj.header = inObject.markers[i].header;

            // get ordered pt 0-7
            std::vector<geometry_msgs::Point> corners_pt;
            GetCorners(inObject.markers[i].points, corners_pt);
            
            obj.pose.position = GetObjCenter(corners_pt);
            obj.pose.orientation = GetObjOrientation(corners_pt);
            obj.dimensions = GetObjDim(corners_pt);
            std::string label_class = inObject.markers[i].text;
            size_t find_idx = label_class.rfind(":");
            if (find_idx != std::string::npos)
            {
                obj.label = label_class.substr(0, find_idx);
                obj.score = std::stof(label_class.substr(find_idx+1));
            }
            std::chrono::system_clock::time_point _convex_start = std::chrono::system_clock::now();
            obj.convex_hull = GetObjHull(corners_pt);
            std::chrono::system_clock::time_point _convex_end = std::chrono::system_clock::now();
            float convex = std::chrono::duration_cast<std::chrono::microseconds>(_convex_end-_convex_start).count();
            total_convex_time += convex;
            
            obj.id = i;
            std::chrono::system_clock::time_point _cloud_start = std::chrono::system_clock::now();
            obj.pointcloud = GetCloud(corners_pt, inCloudPtr, obj.convex_hull);
            std::chrono::system_clock::time_point _cloud_end = std::chrono::system_clock::now();
            float cloud = std::chrono::duration_cast<std::chrono::microseconds>(_cloud_end-_cloud_start).count();
            total_cloud_time += cloud;
            // std::cout << "\033[1;40m"<< cloud << " us\033[0m\n";

            // filtered det without obj
            if( obj.pointcloud.width * obj.pointcloud.height == 0)
            {
                std::cout << "Empty cloud: " << obj.label << ", score: " << obj.score << std::endl;
                continue;
            }

            outMsg.objects.push_back(obj);
        }
        std::cout << "\033[1;40m total cloud time: "<< total_cloud_time << " us\033[0m\n";
        std::cout << "\033[1;40m total convex time: "<< total_convex_time << " us\033[0m\n";
        std::cout << "Total obj: " << objs_size << ", remain: " << outMsg.objects.size() << std::endl;
    }

    void ObjectFilter::SearchCurrentCloud(
        const std_msgs::Header & inHeader,
        sensor_msgs::PointCloud2 & outCloud)
    {
        std::cout << "Search cloud, " << inHeader.stamp << std::endl;
        sensor_msgs::PointCloud2 current_cloud;
        float time_diff = std::fabs(mCloudQueue.front().header.stamp.toSec() - inHeader.stamp.toSec());
        while (mCloudQueue.size() > 0 && 
               time_diff > 0.05 && 
               mCloudQueue.front().header.stamp < inHeader.stamp)
        {
            std::cout << "erase " << mCloudQueue.front().header.stamp <<std::endl;
            mCloudQueue.erase(mCloudQueue.begin());
            time_diff = std::fabs(mCloudQueue.front().header.stamp.toSec() - inHeader.stamp.toSec());
        }

        if(mCloudQueue.size() > 0)
        {
            current_cloud = mCloudQueue.front();
        }
        outCloud = current_cloud;
    }

}

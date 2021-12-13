#include <livox_object_filter/object_filter.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "livox_object_filter_node");
    ObjectFilter::ObjectFilter obj;
    ros::spin();
}

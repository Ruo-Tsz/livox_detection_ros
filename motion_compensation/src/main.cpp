# include "motion_compensation/motion_compensation.h"


int main(int argc, char** argv)
{
    ros::init(argc, argv, "motion_compensation_node");

    MotionCompensation mMotionCompensation;
    mMotionCompensation.Run();

    return 0;
}
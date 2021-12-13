#include <cloud2voxel.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "cloud2voxel");

    Cloud2Voxel::Cloud2Voxel cloud2voxel;
	cloud2voxel.Run();

    return 0;
}

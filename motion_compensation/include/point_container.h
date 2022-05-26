#ifndef POINT_CONTAINER
#define POINT_CONTAINER
#include <iostream>

class PointContainer 
{
    public:
        inline bool IsEmpty() const { return _points.empty(); }
        inline std::vector<int>& points() { return _points; }
        inline const std::vector<int>& points() const { return _points; }
        inline int& state() { return _state; }
        inline const int& state() const { return _state; }
        inline std::vector<int>& obj_label() { return _obj_label; }
        inline int& id() { return _id; }
        inline const int& id() const { return _id; }
        inline float& prob() { return _prob; }
        inline const float& prob() const { return _prob; }
        inline int& ptExist() { return _ptExist; }
        inline const int& ptExist() const { return _ptExist; }

    private:
        std::vector<int> _points;
        int _state;
        std::vector<int> _obj_label;
        int _id;
        float _prob;
        int _ptExist;
};
                  
typedef std::vector<PointContainer> GridSlice;

#endif
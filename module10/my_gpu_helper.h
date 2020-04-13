#ifndef my_gpu_helper_h
#define my_gpu_helper_h

#include <chrono>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> HR_TimePoint;

int get_duration_ns( const HR_TimePoint& start,
                     const HR_TimePoint& end )
{
    return int(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

HR_TimePoint get_clock_time()
{
    return std::chrono::high_resolution_clock::now();
}

#endif //my_gpu_helper_h

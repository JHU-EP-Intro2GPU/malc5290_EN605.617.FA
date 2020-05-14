#ifndef performance_helper_h
#define performance_helper_h

#include <chrono>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> HR_TimePoint;

double get_duration_seconds( const HR_TimePoint& start, const HR_TimePoint& end )
{
    std::chrono::duration<double> dur = end - start;
    return dur.count();
}

HR_TimePoint get_clock_time()
{
    return std::chrono::high_resolution_clock::now();
}

#endif //performance_helper_h

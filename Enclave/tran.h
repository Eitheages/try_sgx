#ifndef TRANS
#define TRANS

#include <vector>

extern "C" int printf(const char* fmt, ...);

std::vector<float> foo1();

extern "C" int foo2(const std::vector<float>& nums);

#endif
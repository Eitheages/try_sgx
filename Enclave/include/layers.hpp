#ifndef NEW_L
#define NEW_L
// #define USEREORDER

#include "layers_bwd.hpp"
#include "layers_fwd.hpp"

// hyper parameter
extern const int N;
extern const int step;
extern float LearningRate;
extern const size_t epoch;
extern const int test_cnt;
extern const memory::dims inputs;
extern std::vector<primitive> net_fwd, net_bwd, net_sgd;
extern std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_args,
    net_sgd_args;
extern dnnl::engine eng;
extern dnnl::stream s;

const int N = 1;  // batch_size
const int step = 1000;
float LearningRate =
    0.01;  // replace with a lower number after serveral training
const size_t epoch = 30;
const int test_cnt = 500;
const memory::dims inputs = {1, 28, 28};  // net input pic
std::vector<primitive> net_fwd{}, net_bwd{}, net_sgd{};
std::vector<std::unordered_map<int, memory>> net_fwd_args{}, net_bwd_args{},
    net_sgd_args{};
engine eng = engine(parse_engine_kind(1, NULL), 0);
stream s(eng);

#endif
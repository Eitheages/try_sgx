#include "../Enclave_t.h"
#include "../include/layers.hpp"

extern "C" int layer_1(float* src, float* dst) {
    // read date (conv1_src) and then calc
    try {
        memory::dims conv1_src_tz = {N, inputs[0], inputs[1], inputs[2]};

        auto conv1_src_memory =
            memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);

        Conv2D conv1(N, conv1_src_tz[3], 6, 5, 1, 2, 0, conv1_src_memory,
                     net_fwd, net_fwd_args, eng);

        write_to_dnnl_memory(src, conv1_src_memory);
        net_fwd.back().execute(s, net_fwd_args.back());

        read_from_dnnl_memory(dst, conv1.arg_dst);
        printf("Intel(R) DNNL: layer1 forward: passed\n");
    } catch (error& e) {
        printf("Intel(R) DNNL: layer1 forward: failed!!!\n");
    }
    return 0;
}

extern "C" int layer_2(float* src, float* dst) {
    // read date (conv1_dst) and then
    try {
        memory::dims conv1_dst_tz = {N, 6, 28, 28};

        auto conv1_dst_memory =
            memory({{conv1_dst_tz}, dt::f32, tag::nchw}, eng);

        Eltwise sigmoid1(algorithm::eltwise_logistic, 0.f, 0.f,
                         conv1_dst_memory, net_fwd, net_fwd_args, eng);

        write_to_dnnl_memory(src, conv1_dst_memory);
        net_fwd.back().execute(s, net_fwd_args.back());

        read_from_dnnl_memory(dst, sigmoid1.arg_dst);
        printf("Intel(R) DNNL: layer2 forward: passed\n");
    } catch (error& e) {
        printf("Intel(R) DNNL: layer2 forward: failed!!!\n");
    }
    return 0;
}
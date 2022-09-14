// #include <opencv2/opencv.hpp>
#include "Enclave_t.h"
#include "layers.hpp"
// #include "mnist_reader.hpp"
// #define DEBUG

// const std::string MNIST_DATA_LOCATION =
//     "/home/aisig/work/mnist-fashion/data/mnist";
// const std::string MNIST_FASHION_DATA_LOCATION =
//     "/home/aisig/work/mnist-fashion/data/fashion";

// typedef mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> type_dataset;

// hyper parameter
const int N = 1;  // batch_size
const int step = 1000;
float LearningRate =
    0.01;  // replace with a lower number after serveral training
const size_t epoch = 30;
const int test_cnt = 500;
const memory::dims inputs = {1, 28, 28};  // net input pic

std::vector<float> train_loss(epoch);
std::vector<float> train_accuracy(epoch);
std::vector<float> test_loss(epoch);
std::vector<float> test_accuracy(epoch);

// void get_data(const type_dataset& dataset, int& t, std::vector<float>& src,
//               std::vector<float>& dst,
//               const memory::dims& resized = {1, 28, 28},
//               std::string data_type = "train", bool normalized = false);

void LeNet(engine::kind engine_kind) {

    // if (60000 < N * step || 10000 < test_cnt) {
    //     std::cout << "there are not so much data!" << std::endl;
    //     return;
    // }

    // get mnist
    // type_dataset dataset =
    //     mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
    //         MNIST_FASHION_DATA_LOCATION, N * step, test_cnt);

    // std::cout << "Train data size: " << dataset.training_images.size() << " "
    //           << "Test data size: " << dataset.test_images.size() << std::endl;

    // Vectors of input data and expected output
    std::vector<float> net_src(N * product(inputs));
    std::vector<float> net_dst(N * 10);  // 10 classes

    auto eng = engine(engine_kind, 0);
    stream s(eng);

    // Vector of primitives and their execute arguments
    std::vector<primitive> net_fwd, net_bwd, net_sgd;
    std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_args,
        net_sgd_args;

    // LeNet conv1
    // {batch, 1, 28, 28} (x) {6, 1, 5, 5} -> {batch, 6, 28, 28}
    // padding = "same"
    memory::dims conv1_src_tz = {N, inputs[0], inputs[1], inputs[2]};

    auto conv1_user_src_memory =
        memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);
    auto conv1_src_memory = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);

    Conv2D conv1(N, conv1_src_tz[3], 6, 5, 1, 2, 0, conv1_src_memory, net_fwd,
                 net_fwd_args, eng);
    Eltwise sigmoid1(algorithm::eltwise_logistic, 0.f, 0.f, conv1.arg_dst,
                     net_fwd, net_fwd_args, eng);

    // LeNet pool1
    // {batch, 6, 28, 28} -> {batch, 6, 14, 14}
    // kernel = {2, 2}, strides = {2, 2}, padding = "valid"

    MeanPool2D pool1(2, 2, sigmoid1.arg_dst, net_fwd, net_fwd_args, eng);

    // LeNet conv2
    // {batch, 6, 14, 14} (x) {16, 6, 5, 5} -> {batch, 16, 10, 10}
    // padding = "valid"
    Conv2D conv2(N, 10, 16, 5, 1, 0, 0, pool1.arg_dst, net_fwd, net_fwd_args,
                 eng);
    Eltwise sigmoid2(algorithm::eltwise_logistic, 0.f, 0.f, conv2.arg_dst,
                     net_fwd, net_fwd_args, eng);

    // LeNet pool2
    // {batch, 16, 10, 10} -> {batch, 16, 5, 5}
    // kernel = {2, 2}, strides = {2, 2}, padding = "valid"

    MeanPool2D pool2(2, 2, sigmoid2.arg_dst, net_fwd, net_fwd_args, eng);

    // LeNet fc1
    // {batch, 16, 5, 5} -> {batch, 120}
    Dense fc1(120, pool2.arg_dst, net_fwd, net_fwd_args, eng);
    Eltwise sigmoid3(algorithm::eltwise_logistic, 0.f, 0.f, fc1.arg_dst,
                     net_fwd, net_fwd_args, eng);

    // LeNet fc2
    // {batch, 120} -> {batch, 84}
    Dense fc2(84, sigmoid3.arg_dst, net_fwd, net_fwd_args, eng);
    Eltwise sigmoid4(algorithm::eltwise_logistic, 0.f, 0.f, fc2.arg_dst,
                     net_fwd, net_fwd_args, eng);

    // LeNet fc3
    // {batch, 84} -> {batch, 10}
    Dense fc3(10, sigmoid4.arg_dst, net_fwd, net_fwd_args, eng);

    // LeNet softmax
    memory::dims softmax_src_tz = {N, 10};
    auto softmax_src_md = memory::desc(softmax_src_tz, dt::f32, tag::nc);
    auto softmax_dec =
        softmax_forward::desc(prop_kind::forward_training, softmax_src_md, 1);
    auto softmax_pd = softmax_forward::primitive_desc(softmax_dec, eng);
    auto softmax_dst_memory = memory(softmax_pd.dst_desc(), eng);

    net_fwd.push_back(softmax_forward(softmax_pd));
    net_fwd_args.push_back(
        {{DNNL_ARG_SRC, fc3.arg_dst}, {DNNL_ARG_DST, softmax_dst_memory}});

    // 0) Clip y_hat to avoid performing log(0)

    float lower = 1e-7;      // alpha
    float upper = 1 - 1e-7;  // beta

    memory::dims y_tz = {N, 10};
    auto y_md = memory::desc({y_tz}, dt::f32, tag::nc);
    // auto y_hat_clipped_memory = memory(y_md, eng);

    // auto clip_desc =
    //     eltwise_forward::desc(prop_kind::forward_training,
    //                           algorithm::eltwise_clip, y_md, lower, upper);
    // auto clip_pd = eltwise_forward::primitive_desc(clip_desc, eng);

    // net_fwd.push_back(eltwise_forward(clip_pd));
    // net_fwd_args.push_back({{DNNL_ARG_SRC, softmax_dst_memory},
    //                         {DNNL_ARG_DST, y_hat_clipped_memory}});

    // // 1) Perform elementwise log on y_hat_cliped
    // auto y_hat_logged_memory = memory(y_md, eng);

    // auto log_desc = eltwise_forward::desc(prop_kind::forward_training,
    //                                       algorithm::eltwise_log, y_md);
    // auto log_pd = eltwise_forward::primitive_desc(log_desc, eng);

    // net_fwd.push_back(eltwise_forward(log_pd));
    // net_fwd_args.push_back({{DNNL_ARG_SRC, y_hat_clipped_memory},
    //                         {DNNL_ARG_DST, y_hat_logged_memory}});
    std::vector<float> y_hat(product(y_tz));
    std::vector<float> y_hat_clipped(product(y_tz));
    std::vector<float> y_hat_logged(product(y_tz));

    // using log(y_hat) and y_true to calculate cross entropy
    // wait until training

    //-----------------------------------------------------------------------
    //----------------- Backpropagation Stream-------------------------------------
    std::vector<float> diff_softmax_src(product(y_tz));
    auto diff_softmax_src_md = memory::desc({y_tz}, dt::f32, tag::nc);
    auto diff_softmax_src_memory = memory(diff_softmax_src_md, eng);

    // softmax back
    // auto softmax_back_desc =
    //     softmax_backward::desc(diff_y_hat_md, softmax_src_md, 1);
    // auto softmax_back_pd =
    //     softmax_backward::primitive_desc(softmax_back_desc, eng, softmax_pd);
    // auto softmax_diff_src_memory = memory(softmax_src_md, eng);

    // net_bwd.push_back(softmax_backward(softmax_back_pd));
    // net_bwd_args.push_back({{DNNL_ARG_DIFF_DST, diff_y_hat_memory},
    //                         {DNNL_ARG_DST, softmax_dst_memory},
    //                         {DNNL_ARG_DIFF_SRC, softmax_diff_src_memory}});

    // fc3 back
    Dense_back_data fc3_back_data(diff_softmax_src_memory, fc3, net_bwd,
                                  net_bwd_args, eng);

    // fc2 sigmoid back
    Eltwise_back sigmoid4_back(algorithm::eltwise_logistic, 0.f, 0.f, sigmoid4,
                               fc3_back_data.arg_diff_src, net_bwd,
                               net_bwd_args, eng);

    // fc2 back
    Dense_back_data fc2_back_data(sigmoid4_back.arg_diff_src, fc2, net_bwd,
                                  net_bwd_args, eng);

    // fc1 sigmoid back
    Eltwise_back sigmoid3_back(algorithm::eltwise_logistic, 0.f, 0.f, sigmoid3,
                               fc2_back_data.arg_diff_src, net_bwd,
                               net_bwd_args, eng);

    // fc1 back
    Dense_back_data fc1_back_data(sigmoid3_back.arg_diff_src, fc1, net_bwd,
                                  net_bwd_args, eng);

    // pool2 back
    MeanPool2D_back pool2_back(2, 2, pool2, fc1_back_data.arg_diff_src, net_bwd,
                               net_bwd_args, eng);

    // conv2 back
    Eltwise_back sigmoid2_back(algorithm::eltwise_logistic, 0.f, 0.f, sigmoid2,
                               pool2_back.arg_diff_src, net_bwd, net_bwd_args,
                               eng);

    Conv2D_back_data conv2_back_data(sigmoid2_back.arg_diff_src, conv2, 1, 0, 0,
                                     net_bwd, net_bwd_args, eng);

    // pool1 back
    MeanPool2D_back pool1_back(2, 2, pool1, conv2_back_data.arg_diff_src,
                               net_bwd, net_bwd_args, eng);

    // conv1 back
    Eltwise_back sigmoid1_back(algorithm::eltwise_logistic, 0.f, 0.f, sigmoid1,
                               pool1_back.arg_diff_src, net_bwd, net_bwd_args,
                               eng);

    Conv2D_back_data conv1_back_data(sigmoid1_back.arg_diff_src, conv1, 1, 2, 0,
                                     net_bwd, net_bwd_args, eng);

    //-----------------------------------------------------------------------
    //----------------- Weights update -------------------------------------
    Conv2D_back_weights conv1_back_weights(sigmoid1_back.arg_diff_src, conv1, 1,
                                           2, 0, net_sgd, net_sgd_args, eng);
    Conv2D_back_weights conv2_back_weights(sigmoid2_back.arg_diff_src, conv2, 1,
                                           0, 0, net_sgd, net_sgd_args, eng);
    Dense_back_weights fc1_back_weights(sigmoid3_back.arg_diff_src, fc1,
                                        net_sgd, net_sgd_args, eng);
    Dense_back_weights fc2_back_weights(sigmoid4_back.arg_diff_src, fc2,
                                        net_bwd, net_bwd_args, eng);
    Dense_back_weights fc3_back_weights(diff_softmax_src_memory, fc3, net_bwd,
                                        net_bwd_args, eng);

    updateWeights_SGD(conv1.arg_weights, conv1_back_weights.arg_diff_weights,
                      LearningRate, net_sgd, net_sgd_args, eng);
    updateWeights_SGD(conv2.arg_weights, conv2_back_weights.arg_diff_weights,
                      LearningRate, net_sgd, net_sgd_args, eng);
    updateWeights_SGD(fc1.arg_weights, fc1_back_weights.arg_diff_weights,
                      LearningRate, net_sgd, net_sgd_args, eng);
    updateWeights_SGD(fc2.arg_weights, fc2_back_weights.arg_diff_weights,
                      LearningRate, net_sgd, net_sgd_args, eng);
    updateWeights_SGD(fc3.arg_weights, fc3_back_weights.arg_diff_weights,
                      LearningRate, net_sgd, net_sgd_args, eng);

    updateWeights_SGD(conv1.arg_bias, conv1_back_weights.arg_diff_bias,
                      LearningRate, net_sgd, net_sgd_args, eng);
    updateWeights_SGD(conv2.arg_bias, conv2_back_weights.arg_diff_bias,
                      LearningRate, net_sgd, net_sgd_args, eng);
    updateWeights_SGD(fc1.arg_bias, fc1_back_weights.arg_diff_bias,
                      LearningRate, net_sgd, net_sgd_args, eng);
    updateWeights_SGD(fc2.arg_bias, fc2_back_weights.arg_diff_bias,
                      LearningRate, net_sgd, net_sgd_args, eng);
    updateWeights_SGD(fc3.arg_bias, fc3_back_weights.arg_diff_bias,
                      LearningRate, net_sgd, net_sgd_args, eng);

    // data index

    // --------------------- training ----------------------------
    int test_step = test_cnt / N;
    std::vector<float> entropy(std::max(step, test_step));

    for (size_t k = 0; k < epoch; ++k) {
        if ((k + 1) % 10 == 0)  // update date learning rate every 10 training
            LearningRate /= 2;

        printf("%-3dth training...\n", k + 1);
        // std::cout << std::left << std::setw(3) << k + 1 << "th training..."
        //           << std::endl;
        int train_t = 0;
        int test_t = 0;
        int train_accurate_time = 0;

        // step
        for (int g = 0; g < step; ++g) {

            // get src and dst
            // get_data(dataset, train_t, net_src, net_dst, inputs, "train",
            //          false);

            // --------------------------------------
            for (int i = 0; i < N; ++i) {
                float* img_arr = new float[784];
                float* img_label = new float[10];
                get_data(img_arr, img_label, N, N * g + i, 0);

                for (int j = 0; j < 784; ++j)
                    net_src[j + 784 * i] = img_arr[j];
                for (int j = 0; j < 10; ++j)
                    net_dst[j + 10 * i] = img_label[j];

                delete img_arr, img_label;
            }
            // printf("get data!\n");
            // --------------------------------------
            // for (size_t i = (N - 1) * 784; i < N * 784; ++i)
            //     printf("%.1f\n", net_src[i]);

            write_to_dnnl_memory(net_src.data(), conv1_src_memory, N * 784);
            // std::vector<float> net_src_new(N * 784);
            // read_from_dnnl_memory(net_src_new.data(), conv1_src_memory);
            // ---------------------------------------------------------------
            // forward calculate
            for (int i = 0; i < net_fwd.size(); ++i) {
                // printf("debug\n");
                net_fwd.at(i).execute(s, net_fwd_args.at(i));
            }

            // calculate the loss function -- cross entropy
            read_from_dnnl_memory(y_hat.data(), softmax_dst_memory);
            // read_from_dnnl_memory(y_hat_logged.data(), y_hat_logged_memory);

            // perform clip and log
            for (size_t j = 0; j < y_hat_logged.size(); ++j) {
                if (y_hat[j] < lower)
                    y_hat_clipped[j] = lower;
                else if (y_hat[j] > upper)
                    y_hat_clipped[j] = upper;
                else
                    y_hat_clipped[j] = y_hat[j];

                y_hat_logged[j] = std::log(y_hat_clipped[j]);
            }

            // three things are done below: calc loss, test accuracy, calc diff_softmax_src
            float crossEntropy = 0;
            size_t infer_ans = 0;
            for (size_t j = 0; j < y_hat_logged.size(); ++j) {
                if (y_hat[infer_ans] < y_hat[j])
                    infer_ans = j;
                if ((j + 1) % 10 == 0) {
                    if (net_dst[infer_ans] > 0.5)
                        ++train_accurate_time;
                    infer_ans = j + 1;
                }
                crossEntropy += y_hat_logged[j] * net_dst[j];
                diff_softmax_src[j] =
                    (y_hat[j] - net_dst[j]);  // maybe N is not neccessary
            }
            crossEntropy /= (float)(-N);
            entropy[g] = crossEntropy;
            write_to_dnnl_memory(diff_softmax_src.data(),
                                 diff_softmax_src_memory, N * 10);

            // backward calculate
            for (size_t i = 0; i < net_bwd.size(); ++i)
                net_bwd.at(i).execute(s, net_bwd_args.at(i));

            // finally update weights and bias
            for (size_t i = 0; i < net_sgd.size(); ++i)
                net_sgd.at(i).execute(s, net_sgd_args.at(i));

            // [for_debug]
            // auto check_memory = conv1_back_weights.arg_diff_weights;
            // std::vector<float> check_data(
            //     product(check_memory.get_desc().dims()));
            // read_from_dnnl_memory(check_data.data(), check_memory);
            // while (1)
            //     break;
            // [for_debug]
        }

        // calc avg_loss (train)
        float train_loss_sum = 0;
        for (size_t g = 0; g < step; ++g)
            train_loss_sum += entropy[g];
        train_loss[k] = train_loss_sum / step;

        // test the model
        int test_accurate_time = 0;
        for (int test_g = 0; test_g < test_step; ++test_g) {

            // get src and dst
            // get_data(dataset, test_t, net_src, net_dst, inputs, "test", false);
            for (int i = 0; i < N; ++i) {
                float* img_arr = new float[784];
                float* img_label = new float[10];
                get_data(img_arr, img_label, N, N * test_g + i, 0);

                for (int j = 0; j < 784; ++j)
                    net_src[j + 784 * i] = float(img_arr[j]);
                for (int j = 0; j < 10; ++j)
                    net_dst[j + 10 * i] = float(img_label[j]);
                delete img_arr, img_label;
            }
            write_to_dnnl_memory(net_src.data(), conv1_src_memory, N * 784);

            // forward calculate (inference)
            for (size_t i = 0; i < net_fwd.size(); ++i)
                net_fwd.at(i).execute(s, net_fwd_args.at(i));
            read_from_dnnl_memory(y_hat.data(), softmax_dst_memory);
            // read_from_dnnl_memory(y_hat_logged.data(), y_hat_logged_memory);

            for (size_t j = 0; j < y_hat_logged.size(); ++j) {
                if (y_hat[j] < lower)
                    y_hat_clipped[j] = lower;
                else if (y_hat[j] > upper)
                    y_hat_clipped[j] = upper;
                else
                    y_hat_clipped[j] = y_hat[j];

                y_hat_logged[j] = std::log(y_hat_clipped[j]);
            }

            // calc test_loss and test_accuracy
            float crossEntropy = 0;
            size_t infer_ans = 0;
            for (size_t j = 0; j < y_hat_logged.size(); ++j) {
                if (y_hat[infer_ans] < y_hat[j])
                    infer_ans = j;
                if ((j + 1) % 10 == 0) {
                    if (net_dst[infer_ans] > 0.5)
                        ++test_accurate_time;
                    infer_ans = j + 1;
                }
                crossEntropy += y_hat_logged[j] * net_dst[j];
            }
            crossEntropy /= (float)(-N);
            entropy[test_g] = crossEntropy;
        }

        // calc avg_test_loss
        float test_loss_sum = 0;
        for (size_t test_g = 0; test_g < test_step; ++test_g)
            test_loss_sum += entropy[test_g];
        test_loss[k] = train_loss_sum / test_step;
        train_accuracy[k] = train_accurate_time / (float)(N * step);
        test_accuracy[k] = test_accurate_time / (float)(test_cnt);

        // std::cout << std::left << std::setw(3) << k + 1
        //           << "train loss: " << std::fixed << std::setprecision(3)
        //           << train_loss[k] << ", train accuracy: " << std::fixed
        //           << std::setprecision(3)
        //           << train_accurate_time / (float)(N * step) << std::endl;
        // std::cout << std::left << std::setw(3) << ""
        //           << "test loss: " << std::fixed << std::setprecision(3)
        //           << test_loss[k] << ", test accuracy: " << std::fixed
        //           << std::setprecision(3)
        //           << test_accurate_time / (float)(test_cnt) << std::endl;
        // std::cout << std::endl;
        printf("%-3dtrain loss: %.3f, train accuracy: %.3f\n", k + 1,
               train_loss[k], train_accuracy[k]);
        printf("   test loss: %.3f, test accuracy: %.3f\n", test_loss[k],
               test_accuracy[k]);
    }
    // for (size_t k = 0; k < epoch; ++k)
    //     std::cout << k << "th training, loss: " << loss[k] << std::endl;
    s.wait();
}

void print_vector(const std::vector<float>& nums) {
    for (size_t i = 0; i < nums.size(); ++i)
        printf("%f\n", nums[i]);
}

extern "C" int try_LeNet() {
    engine::kind engine_kind = parse_engine_kind(1, NULL);  //CPU
    LeNet(engine_kind);
    printf("LeNet passed!\n");
    printf("Train loss:\n");
    print_vector(train_loss);
    printf("Train accuracy:\n");
    print_vector(train_accuracy);
    printf("Test loss:\n");
    print_vector(test_loss);
    printf("Test accuracy:\n");
    print_vector(test_accuracy);

    return 0;
}

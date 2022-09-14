/*
 * Copyright (C) 2011-2021 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

// App.cpp : Define the entry point for the console application.
//

#include <assert.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>
// #include "mnist_reader.hpp"
#include "./Common/mnist_reader.hpp"
#include "Enclave_u.h"
#include "sgx_tseal.h"
#include "sgx_urts.h"

#define ENCLAVE_NAME "libenclave.signed.so"

// Global data
sgx_enclave_id_t global_eid = 0;

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char* msg;
    const char* sug; /* Suggestion */
} sgx_errlist_t;

using namespace std;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {SGX_ERROR_UNEXPECTED, "Unexpected error occurred.", NULL},
    {SGX_ERROR_INVALID_PARAMETER, "Invalid parameter.", NULL},
    {SGX_ERROR_OUT_OF_MEMORY, "Out of memory.", NULL},
    {SGX_ERROR_ENCLAVE_LOST, "Power transition occurred.",
     "Please refer to the sample \"PowerTransition\" for details."},
    {SGX_ERROR_INVALID_ENCLAVE, "Invalid enclave image.", NULL},
    {SGX_ERROR_INVALID_ENCLAVE_ID, "Invalid enclave identification.", NULL},
    {SGX_ERROR_INVALID_SIGNATURE, "Invalid enclave signature.", NULL},
    {SGX_ERROR_OUT_OF_EPC, "Out of EPC memory.", NULL},
    {SGX_ERROR_NO_DEVICE, "Invalid SGX device.",
     "Please make sure SGX module is enabled in the BIOS, and install SGX "
     "driver afterwards."},
    {SGX_ERROR_MEMORY_MAP_CONFLICT, "Memory map conflicted.", NULL},
    {SGX_ERROR_INVALID_METADATA, "Invalid enclave metadata.", NULL},
    {SGX_ERROR_DEVICE_BUSY, "SGX device was busy.", NULL},
    {SGX_ERROR_INVALID_VERSION, "Enclave version was invalid.", NULL},
    {SGX_ERROR_INVALID_ATTRIBUTE, "Enclave was not authorized.", NULL},
    {SGX_ERROR_ENCLAVE_FILE_ACCESS, "Can't open enclave file.", NULL},
    {SGX_ERROR_MEMORY_MAP_FAILURE, "Failed to reserve memory for the enclave.",
     NULL},
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret) {
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist / sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if (ret == sgx_errlist[idx].err) {
            if (NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }

    if (idx == ttl)
        printf(
            "Error code is 0x%X. Please refer to the \"Intel SGX SDK Developer "
            "Reference\" for more details.\n",
            ret);
}

/* OCall functions */
void ocall_print_string(const char* str) {
    /* Proxy/Bridge will check the length and null-terminate
     * the input string to prevent buffer overflow.
     */
    printf("%s", str);
}

// --------------------------------------------
typedef mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>
    type_dataset;

const std::string MNIST_DATA_LOCATION =
    "/home/cauchy/github/mnist-fashion/data/mnist";
const std::string MNIST_FASHION_DATA_LOCATION =
    "/home/cauchy/github/mnist-fashion/data/fashion";
type_dataset dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
        MNIST_DATA_LOCATION, 0, 0);

void get_data(float* src, float* dst, int N, int t, int data_type) {
    // read one batch (N images and N answers), step is this step (0-based)
    for (size_t i = 0; i < N * 10; ++i)
        dst[i] = (float)0;
    std::vector<uint8_t> pic;
    size_t ans;

    if (data_type == 0) {
        pic = dataset.training_images[t];
        ans = dataset.training_labels[t];
    } else if (data_type == 1) {
        pic = dataset.test_images[t];
        ans = dataset.test_labels[t];
    }

    size_t input_size = 784;

    for (size_t j = 0; j < input_size; ++j)
        src[j] = ((float)pic[j]);

    // write data into dst
    dst[ans] = (float)1;
    // printf("123\n");
}
// -----------------------------------------------

int main(int argc, char* argv[]) {
    (void)argc, (void)argv;

    sgx_status_t ret = SGX_SUCCESS;
    int retval = 0;
    sgx_enclave_id_t eid0 = 0;
    sgx_enclave_id_t eid1 = 1;

    // load the enclave
    // Debug: set the 2nd parameter to 1 which indicates the enclave are launched in debug mode
    ret = sgx_create_enclave(ENCLAVE_NAME, SGX_DEBUG_FLAG, NULL, NULL, &eid0,
                             NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        return -1;
    }

    ret = sgx_create_enclave(ENCLAVE_NAME, SGX_DEBUG_FLAG, NULL, NULL, &eid1,
                             NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        return -1;
    }

    // cout << "Intel(R) Deep Neural Network Library (DNNL)" << endl;

    // ret = try_LeNet(eid, &retval);
    // if (ret != SGX_SUCCESS) {
    //     print_error_message(ret);
    // }

    float* src = new float[1 * 1 * 28 * 28];
    float* dst = new float[1 * 6 * 28 * 28];

    ret = layer_1(eid0, &retval, src, dst);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
    }
    float* t = src;
    src = dst;
    delete t;
    dst = static_cast<float*>(malloc(sizeof((float)0) * 1 * 6 * 28 * 28));
    printf("debug!\n");

    ret = layer_2(eid1, &retval, src, dst);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
    }

    // Destroy the enclave
    sgx_destroy_enclave(eid0);
    sgx_destroy_enclave(eid1);

    cout << "Enter a character before exit ..." << endl;
    getchar();
    return 0;
}

//
// Created by stephehuang on 2020/8/30.
//

// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"
namespace TNN_NS {
DECLARE_CPU_ACC(MatMul, LAYER_MAYTMUL);

Status CpuMatMulLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuMatMulLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    // Matrix A: (count, M, K)
    // Matrix B: (count, K, N)
    // Matrix C: (count, M, N)
    auto param         = dynamic_cast<MatMulLayerParam *>(param_);
    int axis           = param->axis;
    auto matrix_a      = inputs[0];
    auto matrix_b      = inputs[1];
    auto matrix_c      = outputs[0];
    auto matrix_a_dims = matrix_a->GetBlobDesc().dims;
    auto matrix_b_dims = matrix_b->GetBlobDesc().dims;
    int count          = DimsVectorUtils::Count(matrix_a_dims, 0, axis);
    int M              = matrix_a_dims[axis];
    int K              = matrix_a_dims[axis + 1];
    int N              = matrix_b_dims[axis + 1];
    if (matrix_a->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        auto matrix_a_ptr = static_cast<float *>(matrix_a->GetHandle().base);
        auto matrix_b_ptr = static_cast<float *>(matrix_b->GetHandle().base);
        auto matrix_c_ptr = static_cast<float *>(matrix_c->GetHandle().base);
        for (int c = 0; c < count; ++c) {
            for (int m = 0; m < M; ++m) {
                float sum = 0;
                for (int n = 0; n < N; ++n) {
                    for (int k = 0; k < K; ++k) {
                        sum += matrix_a_ptr[c * M * K + m * K + k] * matrix_b_ptr[c * K * N + k * N + n];
                    }
                    matrix_c_ptr[c * M * N + m * N + n] = sum;
                }
            }
        }
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(MatMul, LAYER_MATMUL);

}  // namespace TNN_NS

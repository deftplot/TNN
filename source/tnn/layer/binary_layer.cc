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

#include "tnn/layer/binary_layer.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

bool CheckBroadcast(DimsVector matrix_a_dims, DimsVector matrix_b_dims) {
    int matrix_a_size = matrix_a_dims.size();
    int matrix_b_size = matrix_b_dims.size();
    int count         = std::min(matrix_a_size, matrix_b_size);
    for (int i = 1; i <= count; ++i) {
        int dim_a = matrix_a_dims[matrix_a_size - i];
        int dim_b = matrix_b_dims[matrix_b_size - i];
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }
    return true;
}

DimsVector CalculateOutputShape(DimsVector matrix_a_dims, DimsVector matrix_b_dims) {
    int matrix_a_size = matrix_a_dims.size();
    int matrix_b_size = matrix_b_dims.size();
    int count         = std::max(matrix_a_size, matrix_b_size);
    DimsVector matrix_c_dims;
    matrix_c_dims.resize(count);
    for (int i = 1; i <= count; ++i) {
        int dim_a                = matrix_a_size - i >= 0 ? matrix_a_dims[matrix_a_size - i] : 1;
        int dim_b                = matrix_b_size - i >= 0 ? matrix_b_dims[matrix_b_size - i] : 1;
        matrix_c_dims[count - i] = std::max(dim_a, dim_b);
    }
    return matrix_c_dims;
}

DimsVector CalculateWeightShape(DimsVector matrix_a_dim, int weight_size) {
    DimsVector weight_dims;
    int n = matrix_a_dim[0];
    int c = matrix_a_dim[1];
    int h = matrix_a_dim[2];
    int w = matrix_a_dim[3];
    if (weight_size == c * h * w) {
        weight_dims = {1, c, h, w};
    } else if (weight_size == c) {
        weight_dims = {1, c, 1, 1};
    } else if (weight_size == h * w) {
        weight_dims = {1, 1, h, w};
    } else if (weight_size == 1) {
        weight_dims = {1, 1, 1, 1};
    } else if (w != 1 && h == 1 && weight_size % w == 0) {
        // for weights shape: {1, 1, h, w}
        //     input   shape: {1, 1, 1, w}
        weight_dims = {1, 1, h, w};
    } else if (h != 1 && c == 1 && weight_size % h == 0) {
        // for weights shape: {1, c, h, 1}
        //     input   shape: {1, 1, h, 1}
        weight_dims = {1, c, h, 1};
    } else if (c != 1 && h == 1 && weight_size % c == 0) {
        // for weights shape: {1, c, h, 1}
        //     input   shape: {1, c, 1, 1}
        weight_dims = {1, c, h, 1};
    }
    return weight_dims;
}

Status BinaryLayer::InferOutputShape() {
    auto layer_resource = dynamic_cast<EltwiseLayerResource *>(resource_);
    auto matrix_a       = input_blobs_[0];
    auto matrix_c       = output_blobs_[0];

    DimsVector matrix_a_dims = matrix_a->GetBlobDesc().dims;
    DimsVector matrix_b_dims;
    if (input_blobs_.size() == 2) {
        auto matrix_b = input_blobs_[1];
        matrix_b_dims = matrix_b->GetBlobDesc().dims;
        if (!CheckBroadcast(matrix_a_dims, matrix_b_dims)) {
            LOGE("Binary InferOutputShape Error\n");
            LOGE("Binary operands could not be broadcast together with shapes (%d, %d, %d, %d) vs (%d, %d, %d, %d)\n",
                 matrix_a_dims[0], matrix_a_dims[1], matrix_a_dims[2], matrix_a_dims[3], matrix_b_dims[0],
                 matrix_b_dims[1], matrix_b_dims[2], matrix_b_dims[3]);
            return TNNERR_MODEL_ERR;
        }
    } else if (input_blobs_.size() == 1 && layer_resource != nullptr) {
        matrix_a_dims    = matrix_a->GetBlobDesc().dims;
        auto weight_size = layer_resource->element_handle.GetDataCount();
        matrix_b_dims    = CalculateWeightShape(matrix_a_dims, weight_size);
        if (matrix_b_dims.empty()) {
            LOGE("Binary InferOutputShape Error, we can not calculate weight shape.\n");
            return TNNERR_MODEL_ERR;
        }
        layer_resource->element_shape =matrix_b_dims;
        if (!CheckBroadcast(matrix_a_dims, matrix_b_dims)) {
            LOGE("Binary InferOutputShape Error\n");
            LOGE("Binary operands could not be broadcast together with shapes (%d, %d, %d, %d) vs (%d, %d, %d, %d)\n",
                 matrix_a_dims[0], matrix_a_dims[1], matrix_a_dims[2], matrix_a_dims[3], matrix_b_dims[0],
                 matrix_b_dims[1], matrix_b_dims[2], matrix_b_dims[3]);
            return TNNERR_MODEL_ERR;
        }
    }
    matrix_c->GetBlobDesc().dims = CalculateOutputShape(matrix_a_dims, matrix_b_dims);
    return TNN_OK;
}

}  // namespace TNN_NS

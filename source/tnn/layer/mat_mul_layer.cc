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

#include <cmath>

#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
DECLARE_LAYER(MatMul, LAYER_MATMUL);

Status MatMulLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status MatMulLayer::InferOutputShape() {
    // TODO MatMul only support case: A.dims.size equal to B.dims.size
    // other case would be support in future
    ASSERT(input_blobs_.size() == 2);
    auto param = dynamic_cast<MatMulLayerParam*>(param_);
    auto& output_dim = output_blobs_[0]->GetBlobDesc().dims;
    output_dim.resize(4);
    auto matrix_a_dim   = input_blobs_[0]->GetBlobDesc().dims;
    auto matrix_b_dim   = input_blobs_[1]->GetBlobDesc().dims;
    int pad_index = matrix_a_dim.size();
    for (int i = matrix_a_dim.size() - 1; i >= 0; --i) {
        if (matrix_a_dim[i] == matrix_b_dim[i] && matrix_a_dim[i] == 1) {
            pad_index -= 1;
        } else {
            break;
        }
    }
	// input0:[1, 32, 64, 1]
	// input1:[1, 64, 96, 1]
	//                    |
	//                  pad _axis
    switch (pad_index) {
        case 4:{
            ASSERT(matrix_a_dim[0] == matrix_b_dim[0]);
            ASSERT(matrix_a_dim[1] == matrix_b_dim[1]);
            ASSERT(matrix_a_dim[3] == matrix_b_dim[2]);
            param->axis   = 2;
            output_dim[0] = matrix_a_dim[0];
            output_dim[1] = matrix_a_dim[1];
            output_dim[2] = matrix_a_dim[2];
            output_dim[3] = matrix_b_dim[3];
            break;
        }
        case 3: {
            ASSERT(matrix_a_dim[0] == matrix_b_dim[0]);
            ASSERT(matrix_a_dim[2] == matrix_b_dim[1]);
            param->axis   = 1;
            output_dim[0] = matrix_a_dim[0];
            output_dim[1] = matrix_a_dim[1];
            output_dim[2] = matrix_b_dim[2];
            output_dim[3] = 1;
            break;
        }
        case 2: {
            ASSERT(matrix_a_dim[1] == matrix_b_dim[0]);
            param->axis   = 0;
            output_dim[0] = matrix_a_dim[0];
            output_dim[1] = matrix_b_dim[1];
            output_dim[2] = 1;
            output_dim[3] = 1;
            break;
        }
        default:{
            LOGE("MatMul get wrong input dims, please check input's dims\n");
            return TNNERR_PARAM_ERR;
        }
    }
    return TNN_OK;
}

REGISTER_LAYER(MatMul, LAYER_MATMUL);

}  // namespace TNN_NS

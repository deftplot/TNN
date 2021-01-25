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
#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"

#include "tnn/device/x86/acc/Float4.h"
#include "tnn/device/x86/acc/Float8.h"

namespace TNN_NS {

DECLARE_X86_ACC(PRelu, X86_PRELU_OP);

template <typename VEC, int pack>
static void prelu_func(float *input, float *output, const float *slope, DimsVector dims, bool is_channel_shared) {
    auto plane = DimsVectorUtils::Count(dims, 2);

    for (int b = 0; b < dims[0]; b++) {
        for (int c = 0; c < dims[1]; c++) {
            float coef    = is_channel_shared ? slope[0] : slope[c];
            auto input_c  = input + b * DimsVectorUtils::Count(dims, 1) + c * plane;
            auto output_c = output + b * DimsVectorUtils::Count(dims, 1) + c * plane;
            int i         = 0;
            VEC v_zero(0.f);
            VEC v_slope(coef);
            for (; i + pack - 1 < plane; i += pack) {
                VEC v_data = VEC::loadu(input_c + i);
                VEC v_res  = VEC::bsl_clt(v_data, v_zero, v_data * v_slope, v_data);
                VEC::saveu(output_c + i, v_res);
            }

            if (i > 0 && i < plane) {
                i = plane - pack;
                VEC v_data = VEC::loadu(input_c + i);
                VEC v_res  = VEC::bsl_clt(v_data, v_zero, v_data * v_slope, v_data);
                VEC::saveu(output_c + i, v_res);
            }
        }
    }
}

Status X86PReluLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status X86PReluLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PReluLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: PReluLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: PReluLayerParam is nil");
    }

    auto layer_res = dynamic_cast<PReluLayerResource *>(resource_);
    if (!layer_res) {
        LOGE("Error: PReluLayerResource is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: PReluLayerResource is nil");
    }

    const int slope_size     = layer_res->slope_handle.GetBytesSize();
    const DataType data_type = layer_res->slope_handle.GetDataType();

    Blob *input_blob       = inputs[0];
    Blob *output_blob      = outputs[0];
    int channel            = output_blob->GetBlobDesc().dims[1];
    int count              = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    const int channel_size = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);
    if (0 == channel_size) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    auto calc = prelu_func<Float8, 8>;
    if (arch_ == sse42) {
        calc = prelu_func<Float4, 4>;
    }

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        const float *slope_data = layer_res->slope_handle.force_to<float *>();

        float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
        float *output_data = static_cast<float *>(output_blob->GetHandle().base);
        calc(input_data, output_data, slope_data, output_blob->GetBlobDesc().dims, layer_param->channel_shared);
    } else {
        return Status(TNNERR_DEVICE_ACC_DATA_FORMAT_NOT_SUPPORT, "Error: this data type not supported in prelu layer");
    }
    return TNN_OK;
}

REGISTER_X86_ACC(PRelu, LAYER_PRELU);

}  // namespace TNN_NS
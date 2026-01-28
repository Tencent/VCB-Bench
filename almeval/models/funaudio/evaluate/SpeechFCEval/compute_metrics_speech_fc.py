# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils.compute_metrics_bfcl import compute_metrics_bfcl_v1
from utils.compute_metrics_acebench import compute_metrics_acebench_v1


def compute_metrics_example():
    base_folder = "./result/"
    model_name = "fun-audio-chat-s2t"

    dataset_name = "SpeechFC-BFCL-Single"
    test_category = "multiple"
    compute_metrics_bfcl_v1(model_name, dataset_name, test_category, base_folder)

    # dataset_name = "SpeechFC-BFCL-Parallel"
    # test_category = "parallel"
    # compute_metrics_bfcl_v1(model_name, dataset_name, test_category, base_folder)

    # dataset_name = "SpeechFC-SmartInteract"
    # test_category = "multiple"
    # compute_metrics_bfcl_v1(model_name, dataset_name, test_category, base_folder)

    # dataset_name = "SpeechFC-ACEBench-Single"
    # test_category = "multiple"
    # compute_metrics_acebench_v1(model_name, dataset_name, test_category, base_folder)

    # dataset_name = "SpeechFC-ACEBench-Parallel"
    # test_category = "multiple"
    # compute_metrics_acebench_v1(model_name, dataset_name, test_category, base_folder)


if __name__ == '__main__':
    compute_metrics_example()






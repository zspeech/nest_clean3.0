#!/bin/bash
# 比较 NeMo 和 nest_ssl_project 的输出

cd nest_ssl_project

python tools/compare_nemo_nest_outputs.py \
    --nemo_output_dir ./saved_nemo_outputs \
    --nest_output_dir ./saved_nest_outputs \
    --atol 1e-5 \
    --rtol 1e-5 \
    --output_file ./comparison_results.txt


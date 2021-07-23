#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr_1
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}

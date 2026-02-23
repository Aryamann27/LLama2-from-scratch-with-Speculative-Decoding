#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

read -p "Enter the URL from email: " PRESIGNED_URL
echo ""
read -p "Enter the list of models to download without spaces (7B,13B,70B,7B-chat,13B-chat,70B-chat), or press Enter for all: " MODEL_SIZE
TARGET_FOLDER="."             # where all files should end up
mkdir -p ${TARGET_FOLDER}

if [[ $MODEL_SIZE == "" ]]; then
    MODEL_SIZE="7B,13B,70B,7B-chat,13B-chat,70B-chat"
fi

echo "Downloading LICENSE and Acceptable Usage Policy"
curl ${PRESIGNED_URL/'*'/"LICENSE"} -o ${TARGET_FOLDER}"/LICENSE"
curl ${PRESIGNED_URL/'*'/"USE_POLICY.md"} -o ${TARGET_FOLDER}"/USE_POLICY.md"

echo "Downloading tokenizer"
curl ${PRESIGNED_URL/'*'/"tokenizer.model"} -o ${TARGET_FOLDER}"/tokenizer.model"
curl ${PRESIGNED_URL/'*'/"tokenizer_checklist.chk"} -o ${TARGET_FOLDER}"/tokenizer_checklist.chk"
(cd ${TARGET_FOLDER} && md5sum -c tokenizer_checklist.chk)

for m in ${MODEL_SIZE//,/ }
do
    if [[ $m == "7B" ]]; then
        SHARD=0
        MODEL_PATH="llama-2-7b"
    elif [[ $m == "7B-chat" ]]; then
        SHARD=0
        MODEL_PATH="llama-2-7b-chat"
    elif [[ $m == "13B" ]]; then
        SHARD=1
        MODEL_PATH="llama-2-13b"
    elif [[ $m == "13B-chat" ]]; then
        SHARD=1
        MODEL_PATH="llama-2-13b-chat"
    elif [[ $m == "70B" ]]; then
        SHARD=7
        MODEL_PATH="llama-2-70b"
    elif [[ $m == "70B-chat" ]]; then
        SHARD=7
        MODEL_PATH="llama-2-70b-chat"
    fi

    echo "Downloading ${MODEL_PATH}"
    mkdir -p ${TARGET_FOLDER}"/${MODEL_PATH}"

    for s in $(seq 0 ${SHARD})
    do
        f=$(printf "%02d" $s)
        curl --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 --continue ${PRESIGNED_URL/'*'/"${MODEL_PATH}/consolidated.${f}.pth"} -o ${TARGET_FOLDER}"/${MODEL_PATH}/consolidated.${f}.pth"
    done

    curl ${PRESIGNED_URL/'*'/"${MODEL_PATH}/params.json"} -o ${TARGET_FOLDER}"/${MODEL_PATH}/params.json"
    curl ${PRESIGNED_URL/'*'/"${MODEL_PATH}/checklist.chk"} -o ${TARGET_FOLDER}"/${MODEL_PATH}/checklist.chk"
    echo "Checking checksums"
    (cd ${TARGET_FOLDER}"/${MODEL_PATH}" && md5sum -c checklist.chk)
done
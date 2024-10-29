#!/bin/bash

# 원본 폴더와 대상 폴더 지정
SOURCE_FOLDER="02_baseline_code_and_model/Compete_COCO_2/labels/test"
DEST_FOLDER="02_baseline_code_and_model/Compete_COCO_2/labels/train"

# 파일 이동
mv "$SOURCE_FOLDER"/* "$DEST_FOLDER"/

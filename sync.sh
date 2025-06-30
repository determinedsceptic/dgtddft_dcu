#!/bin/bash

SOURCE_DIR="$(readlink -f "$(dirname -- "$0")")"
REMOTE_DIR=dongfang:/public/home/acsa/hyj/DGTDDFT-feat-Distribution2

echo "SYNC ${SOURCE_DIR} to ${REMOTE_DIR}..."

# --exclude=xxx --exclude-from=.gitignore 

rsync -avz "${SOURCE_DIR}/" "${REMOTE_DIR}/"

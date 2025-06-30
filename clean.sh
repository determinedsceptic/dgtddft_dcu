#! /bin/bash

source ./env.sh
echo ${DGDFT_DIR}

rm -rf  ${DGDFT_DIR}/external/yaml-cpp
cd ${DGDFT_DIR}/external/yaml-cpp-0.7.0
rm -rf build/*
cd  ${DGDFT_DIR}/external/lbfgs
make cleanall 
cd  ${DGDFT_DIR}/external/rqrcp
make cleanall 
cd  ${DGDFT_DIR}/external/blopex/blopex_abstract
make clean 
cd  ${DGDFT_DIR}/src
make cleanall 
cd  ${DGDFT_DIR}/examples
make cleanall

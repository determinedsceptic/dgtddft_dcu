#! /bin/bash

source ./env.sh
echo ${DGDFT_DIR}


cd  ${DGDFT_DIR}/external/lbfgs
make cleanall && make
cd  ${DGDFT_DIR}/external/rqrcp
make cleanall && make
cd  ${DGDFT_DIR}/external/blopex/blopex_abstract
make clean && make
cd  ${DGDFT_DIR}/src
make cleanall && make -j
cd  ${DGDFT_DIR}/examples
make cleanall && make pwdft && make dghf

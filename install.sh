cd external
cd lbfgs
make cleanall && make -j
cd ../rqrcp
make cleanall && make -j
cd ../../src
make cleanall && make -j
cd ../examples
make -j  cleanall && make -j dghf pwdft

#!/bin/sh

CXX_FLAGS="-I$PWD/libncnn/include"
LD_FLAGS="-L$PWD/libncnn/lib -lncnn -fopenmp"

g++ -Wall -D_TEST_ bmpfile.c ffmtcnn.cpp $CXX_FLAGS $LD_FLAGS -o test.exe
g++ --shared ffmtcnn.cpp $CXX_FLAGS $LD_FLAGS -o ffmtcnn.dll


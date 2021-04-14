#!/bin/sh

set -e

CXX_FLAGS="-I$PWD/../libncnn/include -ffunction-sections -fPIC"
LD_FLAGS="-L$PWD/../libncnn/lib -lncnn -fopenmp -Wl,-gc-sections -Wl,-strip-all"

case "$1" in
"")
    g++ -Wall -D_TEST_ bmpfile.c ffmtcnn.cpp $CXX_FLAGS $LD_FLAGS -o test
    ;;
clean)
    rm -rf test test.exe out.bmp
    ;;
esac

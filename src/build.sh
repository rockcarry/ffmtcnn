#!/bin/sh

set -e

CXX_FLAGS="-I$PWD/../libncnn/include -ffunction-sections"
LD_FLAGS="-L$PWD/../libncnn/lib -lncnn -fopenmp -Wl,-gc-sections -Wl,-strip-all"

case "$1" in
"")
    g++ -Wall -D_TEST_ bmpfile.c ffmtcnn.cpp $CXX_FLAGS $LD_FLAGS -o test.exe
    g++ --shared ffmtcnn.cpp $CXX_FLAGS $LD_FLAGS -o ffmtcnn.dll
    dlltool -l ffmtcnn.lib -d ffmtcnn.def
	strip *.dll *.exe
    ;;
clean)
    rm -rf *.lib *.dll *.exe
    ;;
esac

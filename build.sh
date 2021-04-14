#!/bin/sh

CXX_FLAGS="-I$PWD/libncnn/include -ffunction-sections"
LD_FLAGS="-L$PWD/libncnn/lib -lncnn -fopenmp -Wl,-gc-sections -Wl,-strip-all"

case "$1" in
"")
    g++ -Wall -D_TEST_ bmpfile.c ffmtcnn.cpp $CXX_FLAGS $LD_FLAGS -o test.exe
    g++ --shared ffmtcnn.cpp $CXX_FLAGS $LD_FLAGS -o ffmtcnn.dll
    dlltool -l ffmtcnn.lib -d ffmtcnn.def
    ;;
clean)
    rm *.lib *.dll *.exe
    ;;
esac

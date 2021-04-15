#!/bin/sh

set -e

CXX_FLAGS="-I$PWD/../libncnn/include -std=c++11 -ffunction-sections -fdata-sections -fPIC"
LD_FLAGS="-L$PWD/../libncnn/lib -lncnn -lpthread -Wl,-gc-sections -Wl,-strip-all"

case "$1" in
"")
    $CXX -Wall -D_TEST_ bmpfile.c ffmtcnn.cpp $CXX_FLAGS $LD_FLAGS -o test
    case "$TARGET_PLATFORM" in
    win32)
        $CXX --shared ffmtcnn.cpp $CXX_FLAGS $LD_FLAGS -o ffmtcnn.dll
        dlltool -l ffmtcnn.lib -d ffmtcnn.def
        $STRIP *.exe *.dll
        ;;
    ubuntu)
        $CXX --shared ffmtcnn.cpp $CXX_FLAGS $LD_FLAGS -o ffmtcnn.so
        $STRIP test *.so
        ;;
    msc33x)
        $CXX --shared ffmtcnn.cpp $CXX_FLAGS $LD_FLAGS -o ffmtcnn.so
        $STRIP test *.so
        ;;
    esac
    ;;
clean)
    rm -rf test *.so *.dll *.exe out.bmp
    ;;
esac

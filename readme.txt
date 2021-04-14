+--------------------------------+
 基于 ncnn + mtcnn 的人脸检测程序
+--------------------------------+

基于腾讯的 ncnn + mtcnn 模型，实现了人脸检测功能

在 ubuntu、msys2 和 msc33x 嵌入式 linux 平台都可以编译通过和使用

目前测试结果还算可以，在 msc33x 平台上检测 312x168 的图片，平均一帧耗时 176.4ms
基本上可以达到 5fps 的检测帧率，可以用到人脸检测和追踪的功能上。


编译和运行
----------
目前已经在三个平台上编译通过：
1. ubuntu - envsetup-for-ubuntu.sh
2. msys2  - envsetup-for-msys2.sh
3. msc33x - envsetup-for-msc33x.sh

编译时需要首先执行对应的 envsetup-xxx.sh 设定环境变量：
source envsetup-for-msys2.sh

编译 libncnn 库：
./build-libncnn.sh

编译 src 下的源代码：
cd src
./build.sh

最终生成 test 程序，可用于人脸检测

./test test.bmp models 32
第一个参数是要检测的图片文件
第二个参数是 mtcnn 模型文件路径
第三个参数是最小的人脸检测大小（以像素为单位）



rockcarry
2021-4-14


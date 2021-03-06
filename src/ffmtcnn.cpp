#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <ncnn/net.h>
#include "bmpfile.h"
#include "ffmtcnn.h"

typedef struct {
    int mindetsize;
    ncnn::Mat image;
    ncnn::Net pnet, rnet, onet;
} MTCNN;

static const float SCORE_THRESHOLD[3] = { 0.8f, 0.8f, 0.8f };
static const float NMS_THRESHOLD  [3] = { 0.5f, 0.6f, 0.7f };

static void load_models(MTCNN *mtcnn, char *path)
{
    char file[256];
    snprintf(file, sizeof(file), "%s/pnet.param", path); mtcnn->pnet.load_param(file);
    snprintf(file, sizeof(file), "%s/pnet.bin"  , path); mtcnn->pnet.load_model(file);
    snprintf(file, sizeof(file), "%s/rnet.param", path); mtcnn->rnet.load_param(file);
    snprintf(file, sizeof(file), "%s/rnet.bin"  , path); mtcnn->rnet.load_model(file);
    snprintf(file, sizeof(file), "%s/onet.param", path); mtcnn->onet.load_param(file);
    snprintf(file, sizeof(file), "%s/onet.bin"  , path); mtcnn->onet.load_model(file);
}

static void free_models(MTCNN *mtcnn)
{
    mtcnn->pnet.clear();
    mtcnn->rnet.clear();
    mtcnn->onet.clear();
}

static bool cmp_score(BBOX a, BBOX b) { return a.score > b.score; }

static void nms(std::vector<BBOX> &dstlist, std::vector<BBOX> &srclist, const float threshold, int min)
{
    if (srclist.empty()) return;
    sort(srclist.begin(), srclist.end(), cmp_score);
    int head, i;
    for (head = 0; head < (int)srclist.size(); ) {
        int x11 = srclist[head].x1;
        int y11 = srclist[head].y1;
        int x12 = srclist[head].x2;
        int y12 = srclist[head].y2;
        dstlist.push_back(srclist[head]);
        for (i = head + 1, head = -1; i < (int)srclist.size(); i++) {
            if (srclist[i].score == 0) continue;
            int x21 = srclist[i].x1;
            int y21 = srclist[i].y1;
            int x22 = srclist[i].x2;
            int y22 = srclist[i].y2;
            int xc1 = x11 > x21 ? x11 : x21;
            int yc1 = y11 > y21 ? y11 : y21;
            int xc2 = x12 < x22 ? x12 : x22;
            int yc2 = y12 < y22 ? y12 : y22;
            int sc  = (xc1 < xc2 && yc1 < yc2) ? (xc2 - xc1) * (yc2 - yc1) : 0;
            int s1  = (x12 - x11) * (y12 - y11);
            int s2  = (x22 - x21) * (y22 - y21);
            int ss  = s1 + s2 - sc;
            float iou;
            if (min) iou = (float)sc / (s1 < s2 ? s1 : s2);
            else     iou = (float)sc / ss;
            if (iou > threshold)  srclist[i].score = 0;
            else if (head == -1) head = i;
        }
        if (head == -1) break;
    }
}

static void refine(std::vector<BBOX> &bboxlist, int width, int height, bool square)
{
    if (bboxlist.empty()) return;
    float maxside = 0;
    for (auto &it : bboxlist) {
        int   bbw = it.x2 - it.x1 + 1;
        int   bbh = it.y2 - it.y1 + 1;
        float x1 = it.x1 + it.regre_coord[0] * bbw;
        float y1 = it.y1 + it.regre_coord[1] * bbh;
        float x2 = it.x2 + it.regre_coord[2] * bbw;
        float y2 = it.y2 + it.regre_coord[3] * bbh;
        if (square) {
            float w = x2 - x1 + 1;
            float h = y2 - y1 + 1;
            maxside = (h > w) ? h : w;
            x1 += (w - maxside) * 0.5f;
            y1 += (h - maxside) * 0.5f;
            it.x2 = lround(x1 + maxside - 1);
            it.y2 = lround(y1 + maxside - 1);
            it.x1 = lround(x1);
            it.y1 = lround(y1);
        }
        if (it.x1 < 0     ) it.x1 = 0;
        if (it.y1 < 0     ) it.y1 = 0;
        if (it.x2 > width ) it.x2 = width  - 1;
        if (it.y2 > height) it.y2 = height - 1;
    }
}

static void run_pnet(MTCNN *mtcnn, std::vector<BBOX> &pnet_bbox_list)
{
    const int   MTCNN_CELL_SIZE = 12;
    const float SCALE_FACTOR    = 0.709f;
    float curfactor = (float)MTCNN_CELL_SIZE / mtcnn->mindetsize;
    float cursize   = (mtcnn->image.w < mtcnn->image.h ? mtcnn->image.w : mtcnn->image.h) * curfactor;
    BBOX  bbox      = {0};

    pnet_bbox_list.clear();
    while (cursize > MTCNN_CELL_SIZE) {
        int curw = (int) ceil(mtcnn->image.w * curfactor);
        int curh = (int) ceil(mtcnn->image.h * curfactor);
        ncnn::Mat in, score, location;
        resize_nearest(mtcnn->image, in, curw, curh);

        ncnn::Extractor ex = mtcnn->pnet.create_extractor();
        ex.set_light_mode(true);
        ex.input  ("data"   , in      );
        ex.extract("prob1"  , score   );
        ex.extract("conv4-2", location);

        std::vector<BBOX> list;
        float *p = score.channel(1);
        for (int row = 0; row < score.h; row++) {
            for (int col = 0; col < score.w; col++) {
                if (*p > SCORE_THRESHOLD[0]) {
                    bbox.score = *p;
                    bbox.x1    = lround((2 * col + 1) / curfactor);
                    bbox.y1    = lround((2 * row + 1) / curfactor);
                    bbox.x2    = lround((2 * col + 1 + MTCNN_CELL_SIZE) / curfactor);
                    bbox.y2    = lround((2 * row + 1 + MTCNN_CELL_SIZE) / curfactor);
                    for (int i = 0; i < 4; i++) bbox.regre_coord[i] = location.channel(i)[row * score.w + col];
                    list.push_back(bbox);
                }
                p++;
            }
        }

        nms(pnet_bbox_list, list, NMS_THRESHOLD[0], 0);
        curfactor *= SCALE_FACTOR;
        cursize   *= SCALE_FACTOR;
    }
}

static void run_rnet(MTCNN *mtcnn, std::vector<BBOX> &rnet_bbox_list, std::vector<BBOX> &pnet_bbox_list)
{
    rnet_bbox_list.clear();
    for (auto &it : pnet_bbox_list) {
        ncnn::Mat img, in;
        copy_cut_border(mtcnn->image, img, it.y1, mtcnn->image.h - it.y2, it.x1, mtcnn->image.w - it.x2);
        resize_nearest(img, in, 24, 24);

        ncnn::Mat score, bbox;
        ncnn::Extractor ex = mtcnn->rnet.create_extractor();
        ex.set_light_mode(true);
        ex.input  ("data"   , in   );
        ex.extract("prob1"  , score);
        ex.extract("conv5-2", bbox );
        if ((float)score[1] > SCORE_THRESHOLD[1]) {
            for (int i = 0; i < 4; i++) it.regre_coord[i] = (float)bbox[i];
            it.score = (float)score[1];
            rnet_bbox_list.push_back(it);
        }
    }
}

static void run_onet(MTCNN *mtcnn, std::vector<BBOX> &onet_bbox_list, std::vector<BBOX> &rnet_bbox_list)
{
    onet_bbox_list.clear();
    for (auto &it : rnet_bbox_list) {
        ncnn::Mat img, in;
        copy_cut_border(mtcnn->image, img, it.y1, mtcnn->image.h - it.y2, it.x1, mtcnn->image.w - it.x2);
        resize_nearest(img, in, 48, 48);

        ncnn::Mat score, bbox, keypoint;
        ncnn::Extractor ex = mtcnn->onet.create_extractor();
        ex.set_light_mode(true);
        ex.input  ("data"   , in      );
        ex.extract("prob1"  , score   );
        ex.extract("conv6-2", bbox    );
        ex.extract("conv6-3", keypoint);
        if ((float)score[1] > SCORE_THRESHOLD[2]) {
            for (int i = 0; i < 4; i++) it.regre_coord[i] = (float)bbox[i];
            it.score = (float)score[1];
            for (int i = 0; i < 5; i++) {
                (it.pointx)[i] = it.x1 + (it.x2 - it.x1) * keypoint[i + 0];
                (it.pointx)[i] = it.y1 + (it.y2 - it.y1) * keypoint[i + 5];
            }
            onet_bbox_list.push_back(it);
        }
    }
}

void* mtcnn_init(char *path, int mindetsize)
{
    MTCNN *mtcnn = new MTCNN();
    if (mtcnn) {
        mtcnn->mindetsize = mindetsize ? mindetsize : 32;
        load_models(mtcnn, path);
    }
    return mtcnn;
}

void mtcnn_free(void *ctxt)
{
    MTCNN *mtcnn = (MTCNN*)ctxt;
    if (mtcnn) {
        free_models(mtcnn);
        delete mtcnn;
    }
}

int mtcnn_detect(void *ctxt, BBOX *bboxlist, int listsize, uint8_t *bitmap, int w, int h)
{
    if (!ctxt || !bitmap) return 0;
    MTCNN *mtcnn = (MTCNN*)ctxt;

    const static float MEAN_VALS[3] = { 127.5, 127.5, 127.5 };
    const static float NORM_VALS[3] = { 0.0078125, 0.0078125, 0.0078125 };
    mtcnn->image = ncnn::Mat::from_pixels(bitmap, ncnn::Mat::PIXEL_RGB, w, h);
    mtcnn->image.substract_mean_normalize(MEAN_VALS, NORM_VALS);

    std::vector<BBOX> temp_bbox_list;
    std::vector<BBOX> pnet_bbox_list;
    std::vector<BBOX> rnet_bbox_list;
    std::vector<BBOX> onet_bbox_list;
    run_pnet(mtcnn, temp_bbox_list);
    if (temp_bbox_list.empty()) return 0;
    nms(pnet_bbox_list, temp_bbox_list, NMS_THRESHOLD[0], 0);
    refine(pnet_bbox_list, mtcnn->image.w, mtcnn->image.h, true);

    run_rnet(mtcnn, temp_bbox_list, pnet_bbox_list);
    if (temp_bbox_list.empty()) return 0;
    nms(rnet_bbox_list, temp_bbox_list, NMS_THRESHOLD[1], 0);
    refine(rnet_bbox_list, mtcnn->image.w, mtcnn->image.h, true);

    run_onet(mtcnn, temp_bbox_list, rnet_bbox_list);
    if (temp_bbox_list.empty()) return 0;
    refine(temp_bbox_list, mtcnn->image.w, mtcnn->image.h, true);
    nms(onet_bbox_list, temp_bbox_list, NMS_THRESHOLD[2], 1);
    mtcnn->image.release();

    int i = 0;
    if (bboxlist) {
        for (i = 0; i < (int)onet_bbox_list.size() && i < listsize; i++) bboxlist[i] = onet_bbox_list[i];
    }
    return i;
}

#ifdef _TEST_
#ifdef WIN32
#define get_tick_count GetTickCount
#else
#include <time.h>
static uint32_t get_tick_count()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
#endif

int main(int argc, char *argv[])
{
    char *bfile = (char*)"test.bmp";
    char *mpath = (char*)"models";
    char *dsize = (char*)"32";
    void *mtcnn = NULL;
    BMP   mybmp = {0};
    BBOX  bboxes[100];
    uint32_t tick;
    int      n, i;

    if (argc >= 2) bfile = argv[1];
    if (argc >= 3) mpath = argv[2];
    if (argc >= 4) dsize = argv[3];
    printf("bmpfile    : %s\n", bfile      );
    printf("models path: %s\n", mpath      );
    printf("detect size: %d\n", atoi(dsize));
    printf("\n");

    mtcnn = mtcnn_init(mpath, atoi(dsize));
    if (0 != bmp_load(&mybmp, bfile)) {
        printf("failed to load bmpfile %s !\n", bfile);
        goto done;
    }

    printf("do face detection 100 times ...\n");
    tick = get_tick_count();
    for (i = 0; i < 100; i++) {
        n = mtcnn_detect(mtcnn, bboxes, 100, (uint8_t*)mybmp.pdata, mybmp.width, mybmp.height);
    }
    printf("finish !\n");
    printf("totoal used time: %d ms\n\n", (int)get_tick_count() - (int)tick);

    printf("face rect list:\n");
    for (i = 0; i < n; i++) {
        printf("score: %.2f, rect: (%3d %3d %3d %3d)\n", bboxes[i].score, bboxes[i].x1, bboxes[i].y1, bboxes[i].x2, bboxes[i].y2);
        bmp_rectangle(&mybmp, bboxes[i].x1, bboxes[i].y1, bboxes[i].x2, bboxes[i].y2, 0, 255, 0);
    }
    printf("\n");

    printf("save result to out.bmp\n");
    bmp_save(&mybmp, (char*)"out.bmp");
done:
    bmp_free(&mybmp);
    mtcnn_free(mtcnn);
    return 0;
}
#endif


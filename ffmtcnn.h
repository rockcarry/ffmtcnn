#ifndef __FFMTCNN_H__
#define __FFMTCNN_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float score;
    float regre_coord[4];
    float pointx[5];
    float pointy[5];
    int x1, y1, x2, y2;
} BBOX;

void* mtcnn_init  (char *path);
void  mtcnn_free  (void *ctxt);
int   mtcnn_detect(void *mtcnn, BBOX *bboxlist, int n, uint8_t *bitmap, int w, int h);

#ifdef __cplusplus
}
#endif

#endif


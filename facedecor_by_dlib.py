#!/usr/bin/env python
# coding=utf-8
import dlib
import cv
import os
import sys
import numpy
from PIL import Image, ImageDraw
import cv2
PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"

SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im,index):
    rects = detector(im, 1)


    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[index]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


    
def transformation_from_points(points1, points2):


    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def read_im_and_landmarks(fname,index):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im,index)

    return im, s

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def jiaozhen(src,dec,model,index):
    im1, landmarks1 = read_im_and_landmarks(model,0)
    im2, landmarks2 = read_im_and_landmarks(src,index)

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                               landmarks2[ALIGN_POINTS])
    warped_mask = warp_im(im2, M, im1.shape)
    cv2.imwrite(dec,warped_mask )


def detect_object(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    rects=detector(im,1)
 
    
#    results = []
    result=[]
    for count in range(len(rects)):
        s1=rects[count].left()
        s2=rects[count].top()
        s3=rects[count].left()+rects[count].width()
        s4=rects[count].top()+rects[count].height()
        result.append((s1,s2,s3,s4))
    return result
def process(infile,dest_file_path,name):
    model='./face.jpg'
    '''在原图上框出头像并且截取每个头像到单独文件夹'''
    try:
        image = cv.LoadImage(infile)
        flag=1
    except:
        flag=0
    if flag:
        faces = detect_object(infile)
   #     im = Image.open(infile)
        save_path = dest_file_path
    else:
        faces=0
#    try:
#       os.mkdir(save_path)
#    except:
#       pass
    if faces:
        try:
#            draw = ImageDraw.Draw(im)
            count = 0
            lists=open("./list.txt",'a+') 
            for f in faces:
                file_name = os.path.join(save_path,name)
                line=file_name+','+str(f[0])+','+str(f[1])+','+str(f[2])+','+str(f[3])+'\n'
                lists.write(line)               
               # print file_name
                jiaozhen(infile,file_name,model,count)	
                count += 1
            lists.close()
        except:
            pass
    else:
        print "Error: cannot detect faces on %s" % infile

if __name__=="__main__":
   process("./1.JPG",'./3.jpg')

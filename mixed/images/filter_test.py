#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from PIL import Image, ImageFilter, ImageOps, ImageEnhance

import os
import ImageUtils

basePath = os.path.dirname(os.path.realpath(__file__))
imagePath = basePath + '/filter_test'

sourceLayer = Image.open(imagePath+"/sourceLayer.jpg")
sourceMask = Image.open(imagePath+"/sourceMask.png")


def createFilteredImages():
    filters = [
        # "BLUR",
        # "CONTOUR",
        # "DETAIL",
        # "EDGE_ENHANCE",
        # "EDGE_ENHANCE_MORE",
        # "EMBOSS",
        # "FIND_EDGES",
        # "SMOOTH",
        # "SMOOTH_MORE",
        # "SHARPEN"
        ]

    # current chain  jpg => bw > EDGE_ENHANCE > invert
    for f in filters:
        bw = ImageUtils.getBlackAndWhite(sourceLayer, 248)
        img = bw.filter(eval("ImageFilter."+f))
        img = ImageOps.invert(img)
        img.save(imagePath+"/out/sourceLayer-"+f+".jpg")

        img2 = sourceMask.filter(eval("ImageFilter."+f))
        img2.save(imagePath+"/out/sourceMask-"+f+".png")


def enhance():
    enhancer = ImageEnhance.Sharpness(sourceLayer)

    for i in range(1):
        factor = i / 4.0
        enhancer.enhance(factor).show("Sharpness %f" % factor)


def gaussianBlurAndBW():
    img2 = sourceMask.filter(ImageFilter.GaussianBlur(radius=3))
    img2 = ImageUtils.getPNGBWMask(img2,200)
    img2.save(imagePath+"/out/sourceMask-gaussian-blur-bw.png")

def sizeUp():
    size = sourceMask.size
    sizeUp =(size[0] +100, size[1] +100)

    img2 = ImageOps.fit(sourceMask, sizeUp)
    img2 = ImageOps.crop(img2, 50)
    img2.save(imagePath+"/out/sourceMask-sizeup.png")

gaussianBlurAndBW()
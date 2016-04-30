#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from PIL import Image
from PIL import ImageOps
from PIL import ImageChops
from PIL import ImageStat

def splitImageInChunks(img, chunkWidth, chunkHeight):
    chunks = []
    imgwidth, imgheight = img.size

    for row in range(0, imgwidth, chunkWidth):
        for col in range(0, imgheight, chunkHeight):
            box = ( row,
                    col,
                    row + chunkWidth,
                    col + chunkHeight )
            # print "cropping %s,%s,%s,%s" % box

            chunks.append(img.crop(box))

    return chunks

def getBlackAndWhite(img):
    gray = img.convert('L')
    bw = gray.point(lambda x: 0 if x<150 else 255)
    return bw

# given a png image returns a b/w image
def getPNGBWMask(img):
    r,g,b,a = img.split()
    rgb_image = Image.merge('L',(a,))
    return rgb_image
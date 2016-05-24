#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from PIL import Image
from PIL import ImageFilter
from images_filters.base_filter import BaseFilter


class EnhancheFilter(BaseFilter):

    def __init__(self, enhanchement):
        allowed_enhanchements = [
            "BLUR",
            "CONTOUR",
            "DETAIL",
            "EDGE_ENHANCE",
            "EDGE_ENHANCE_MORE",
            "EMBOSS",
            "FIND_EDGES",
            "SMOOTH",
            "SMOOTH_MORE",
            "SHARPEN"
        ]
        enhanchement = enhanchement.upper()

        if not enhanchement in allowed_enhanchements:
            raise ValueError("Enhanchement %s not allowed" % (enhanchement))

        self.enhanchement = enhanchement

    def filter(self, image_object):
        if isinstance(image_object, list):
            for i in image_object:
                try:
                    yield i.filter(eval("ImageFilter."+self.enhanchement))
                except Exception, e:
                    raise e
        else:
            try:
                yield image_object.filter(eval("ImageFilter."+self.enhanchement))
            except Exception, e:
                raise e
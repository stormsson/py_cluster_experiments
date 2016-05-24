#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from PIL import Image
# from PIL import ImageFilter
# from PIL import ImageOps
# from PIL import ImageChops
# from PIL import ImageStat
from images_filters.base_filter import BaseFilter


class ResizeFilter(BaseFilter):

    def __init__(self, size):
        if not isinstance(size, tuple):
            raise ValueError("size param must be a 2-sized tuple")

        if len(size) != 2 :
            raise ValueError("size param must be a 2-sized tuple")

        self.size = size

    def filter(self, image_object):
        if isinstance(image_object, list):
            for i in image_object:
                try:
                    yield i.resize(self.size)
                except Exception, e:
                    raise e
        else:
            try:
                yield image_object.resize(self.size)
            except Exception, e:
                raise e
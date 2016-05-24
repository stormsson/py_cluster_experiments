#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# from PIL import Image
# from PIL import ImageFilter
# from PIL import ImageOps
# from PIL import ImageChops
# from PIL import ImageStat

class BaseFilter():

    def filter(self, image_object):
        if isinstance(image_object, list):
            for i in image_object:
                yield i
        else:
            yield image_object
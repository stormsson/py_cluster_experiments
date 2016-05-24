#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from PIL import Image

def FolderProcessor():

    def __init__(self, paths, filter_list):
        self.paths = paths
        self.filter_list = filter_list

    def run(self):
        for file in paths:
            if not isfile(file):
                raise IOError("File %s not found." % (file))

            img = Image.open(file)
            for f in filter_list:
                img = f.filter(img)

            yield img

#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest, os
from PIL import Image

from images_filters.base_filter import BaseFilter


class BaseFilterTest(unittest.TestCase):

    def setUp(self):
        self.instance = BaseFilter()

    def test_generator(self):

        t1 = Image.new("1", (3,2))
        t2 = [ Image.new("1", (3,2)), Image.new("1", (3,2)), Image.new("1", (3,2)) ]

        for r in self.instance.filter(t1):
            self.assertEqual(r, t1)

        result = []
        for r in self.instance.filter(t2):
            result.append(r)

        self.assertEqual(t2, result)
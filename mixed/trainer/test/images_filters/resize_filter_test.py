#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest, os

from images_filters.resize_filter import ResizeFilter


class ResizeFilterTest(unittest.TestCase):

    # def setUp(self):
    #     self.instance = BaseFilter()

    def test_bad_constructor_raises_exception(self):

        with self.assertRaises(ValueError) as context:
            x = ResizeFilter("not_a_tuple")

        self.assertTrue("size param must be a 2-sized tuple" in context.exception.message)

        with self.assertRaises(ValueError) as context:
            x = ResizeFilter((2))

        self.assertTrue("size param must be a 2-sized tuple" in context.exception.message)
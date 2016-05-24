#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest, os

from images_filters.enhanche_filter import EnhancheFilter


class ResizeFilterTest(unittest.TestCase):

    # def setUp(self):
    #     self.instance = BaseFilter()

    def test_bad_constructor_raises_exception(self):

        bad_filter = "NOT_A_VALID_FILTER"
        with self.assertRaises(ValueError) as context:
            x = EnhancheFilter(bad_filter)

        self.assertTrue("Enhanchement %s not allowed" % (bad_filter) in context.exception.message)

#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest, os

from dataset_generator.images_dataset_generator import ImagesDatasetGenerator


class ImagesDatasetGeneratorTest(unittest.TestCase):

    def getSimpleInstance(self, classes_labels=["a", "b"]):
        instance = ImagesDatasetGenerator(2, 2, classes_labels)
        return instance

    def setUp(self):
        self.simpleInstance = self.getSimpleInstance()


    def test_get_classes_labels(self):
        classes_labels = ["a", "b"]
        instance = ImagesDatasetGenerator(2, 2, classes_labels)
        self.assertEqual(classes_labels, instance.getClassesLabels())

    def test_validate_class_label(self):
        with self.assertRaises(LookupError) as context:
            self.simpleInstance.validateClassLabel("c")

        self.assertTrue("class label c not found" in context.exception.message)

    def test_add_folder_of_nonexistent_class(self):
        with self.assertRaises(LookupError) as context:
            self.simpleInstance.addFolder(os.path.dirname(os.path.realpath(__file__)), "c")

        self.assertTrue("class label c not found" in context.exception.message)

    def test_add_folder_of_existent_class(self):
        self.simpleInstance.addFolder(os.path.dirname(os.path.realpath(__file__)), "a")
        self.simpleInstance.addFolder(os.path.dirname(os.path.realpath(__file__)), "a")
        expected_folder_list = {"a":[os.path.dirname(os.path.realpath(__file__))]}
        expected_filters_list = {"a":[]}

        self.assertEqual(expected_folder_list, self.simpleInstance.getFolders())
        self.assertEqual(expected_filters_list, self.simpleInstance.getFilters())

    def test_add_folder_of_unreadable_path(self):

        fake_path = "fake_path"
        with self.assertRaises(IOError) as context:
            self.simpleInstance.addFolder(fake_path, "a")

        self.assertTrue("unreadable path: "+fake_path in context.exception.message)

    def test_set_filters(self):
        fake_filters = [1,2,3]
        self.simpleInstance.setFilters("a", fake_filters)
        expected_filters = {"a" : fake_filters}
        expected_a_filters = fake_filters

        self.assertEqual(expected_filters, self.simpleInstance.getFilters())
        self.assertEqual(expected_a_filters, self.simpleInstance.getFilters("a"))

    def test_fet_filters_with_wrong_class_label_raises_exception(self):

        with self.assertRaises(LookupError) as context:
            self.simpleInstance.getFilters("c")

        self.assertTrue("class label c not found" in context.exception.message)


    def test_set_filters_with_wrong_class_label_raises_exception(self):

        with self.assertRaises(LookupError) as context:
            self.simpleInstance.setFilters("c", [])

        self.assertTrue("class label c not found" in context.exception.message)





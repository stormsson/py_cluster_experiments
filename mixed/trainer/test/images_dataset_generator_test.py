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

    def test_add_folder_of_nonexistent_class(self):
        with self.assertRaises(LookupError) as context:
            self.simpleInstance.addFolder(os.path.dirname(os.path.realpath(__file__)), "c")

        self.assertTrue("class label c not found" in context.exception.message)



    def test_add_folder_of_existent_class(self):
        self.simpleInstance.addFolder(os.path.dirname(os.path.realpath(__file__)), "a")
        self.simpleInstance.addFolder(os.path.dirname(os.path.realpath(__file__)), "a")
        expected_folder_list = {"a":[os.path.dirname(os.path.realpath(__file__))]}

        self.assertEqual(expected_folder_list, self.simpleInstance.getFolders())

    def test_add_folder_of_unreadable_path(self):

        fake_path = "fake_path"
        with self.assertRaises(IOError) as context:
            self.simpleInstance.addFolder(fake_path, "a")

        self.assertTrue("unreadable path: "+fake_path in context.exception.message)





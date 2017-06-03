#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 04 00:46:54 2017

@author: mortza
"""
import unittest


class Tests(unittest.TestCase):

    def test_dummy(self):
        self.assertTrue(1 == 1, msg="sample test.")


if __name__ == '__main__':
    unittest.main()

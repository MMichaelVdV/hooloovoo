import unittest

import numpy as np
import torch
from hooloovoo.utils.imagetools import color_classes, image_as_numpy


class TestImageTools(unittest.TestCase):

    def test_color_n_classes(self):
        y5 = torch.randn((5, 10, 20))  # 5 classes, 10x20 image (torch convention: CHW)
        y5[0, :5, :5] = 100
        img5 = color_classes(y5)

        # from hooloovoo.utils.plotter import img_show
        # img_show(y5[0], img5, wait=True)

        self.assertEqual("RGBA", img5.mode)
        self.assertEqual((20, 10), img5.size)  # PIL convention: WH
        self.assertEqual((10, 20, 4), image_as_numpy(img5).shape)  # numpy RGBA convention: HWC
        self.assertTrue(np.all(image_as_numpy(img5)[:5, :5] == (0, 0, 0, 255)))  # top left corner opaque black

    def test_color_2_classes(self):
        y2 = torch.empty((2, 10, 20))
        y2[0] = 1
        y2[1] = 0
        y2[1, 5:8, 5:15] = 2
        img2 = color_classes(y2)

        # from hooloovoo.utils.plotter import img_show
        # img_show(y2[1], img2, wait=True)

        self.assertEqual("RGBA", img2.mode)
        self.assertEqual((20, 10), img2.size)
        self.assertEqual((10, 20, 4), image_as_numpy(img2).shape)
        self.assertTrue(np.all(image_as_numpy(img2)[5:8, 5:15] == (255, 255, 255, 255)))
        self.assertTrue(np.all(image_as_numpy(img2)[0:5, 0:5] == (0, 0, 0, 255)))

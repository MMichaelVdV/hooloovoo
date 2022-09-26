import os
import numpy as np
from os import path

import torch
from PIL import Image
from torchvision.transforms import functional as f

from hooloovoo.deeplearning.inference.infer_piecewise import infer_piecewise
from hooloovoo.deeplearning.networks.network import Network
from hooloovoo.utils.arbitrary import HW
from hooloovoo.utils.imagetools import image_as_numpy, project_segmentation_contours, image_as_pil, \
    largest_cc_and_neighbours, is_image_filename
from hooloovoo.utils.ostools import makedir
from hooloovoo.utils.plotter import img_show


class Infer:

    softmax = torch.nn.Softmax2d()

    def __init__(self, settings, model: Network):
        self.settings = settings.inference
        self.settings.max_size = HW(**self.settings.max_size.to_dict())
        self.model = model
        self.device = settings.device

        model_path = self.settings.paths.model
        if model_path is not None:
            state = torch.load(model_path)
            self.model.load_state_dict(state)

    def _infer(self, image_path, out_path=None, verbose=True, plot=False, threshold=None, **kwargs):
        in_image = Image.open(image_path)
        x = f.to_tensor(in_image).unsqueeze(0)
        y = infer_piecewise(self.model, x, max_size=self.settings.max_size, process_y_fn=self.softmax,
                            inference_device=self.device, out_device=torch.device('cpu'),
                            verbose=verbose)

        out_image = image_as_numpy(y[0, 1])
        try:
            processed_image = largest_cc_and_neighbours(out_image, threshold=threshold, **kwargs)
        except ValueError:
            processed_image = np.zeros_like(out_image, dtype=np.int)
        projected_outline = project_segmentation_contours(in_image, processed_image, threshold=threshold)

        if plot is True or plot == "mask":
            img_show(in_image, out_image, processed_image, ncol=3, wait=True)
        if plot is True or plot == "outline":
            img_show(projected_outline, wait=True)

        if out_path is not None:
            name, ext = path.splitext(out_path)
            segmentation_raw = image_as_pil(out_image)
            segmentation_raw.save(name + ".probability" + ext)
            segmentation_processed = image_as_pil(processed_image)
            segmentation_processed.save(name + ".mask" + ext)
            segmentation_projection = image_as_pil(projected_outline)
            segmentation_projection.save(name + ".outline" + ext)

    def run(self):
        input_dir = self.settings.paths.data.x
        output_dir = self.settings.paths.data.y
        for x_dir, _, files in os.walk(input_dir):
            rel_path = path.relpath(x_dir, input_dir)
            y_dir = path.join(output_dir, rel_path)
            makedir(y_dir, exists_ok=True, recursive=True)
            for filename in files:
                if is_image_filename(filename):
                    x_path = path.normpath(path.join(x_dir, filename))
                    y_path = path.normpath(path.join(y_dir, filename))
                    print("processing {} -> {}".format(x_path, y_path))
                    self._infer(x_path, y_path,
                                verbose=True, plot=False,
                                include_distance=self.settings.postprocess.include_distance,
                                threshold=self.settings.postprocess.threshold)

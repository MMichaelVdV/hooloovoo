import os
from abc import ABC
from os import path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.nn import ZeroPad2d
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import functional as f

from hooloovoo.deeplearning.networks.network import Network
from hooloovoo.deeplearning.training.augmentation import random_color_jitter_fn, random_mirror_and_quarter_rot_fn
from hooloovoo.deeplearning.training.introspection import print_param_overview
from hooloovoo.deeplearning.training.preprocessing import PreProcess, equal_spread_crop_boxes, PreProcessCache
from hooloovoo.deeplearning.training.train_loop import TrainLoop, Event
from hooloovoo.utils.arbitrary import hex_of_hash, HW
from hooloovoo.utils.functional import flatten, both
from hooloovoo.utils.imagetools import crop_pil_image, image_as_numpy
from hooloovoo.utils.ostools import makedir
from hooloovoo.utils.plotter import img_show
from hooloovoo.utils.ptree import ptree
from hooloovoo.utils.tensortools import align_target
from hooloovoo.utils.timer import time_hms, hms


class Train:
    def __init__(self, settings, model: Network):
        self.settings = settings.training
        self.settings.preprocess.max_size = HW(**self.settings.preprocess.max_size.to_dict())
        self.model: Network = model
        self.device = settings.device

    def get_cache_data(self) -> Tuple[PreProcessCache, PreProcessCache]:

        def load_pair(xy: Tuple[str, str]):
            input_path, target_path = xy
            input_image = Image.open(input_path).convert("RGB")
            target_image = Image.open(target_path).convert("L")
            return input_image, target_image

        def split(x: Image):
            boxes = equal_spread_crop_boxes(
                in_image_size=HW(*x.size[::-1]),
                cropped_image_size=self.settings.preprocess.max_size,
                bidir_overlap=self.settings.preprocess.splits_overlap
            )
            return [crop_pil_image(x, box) for box in boxes]

        def split_single(x: Image):
            return PreProcess.from_list(split(x))

        def split_pair(x: Image, y: Image):
            assert x.size == y.size
            return PreProcess.from_list(list(zip(split(x), split(y))))

        def has_foreground(y: Image) -> bool:
            fraction_foreground = (image_as_numpy(y) > (255 / 2)).mean()
            return fraction_foreground > self.settings.preprocess.min_foreground_fraction

        def save_pair(dir, xy):
            x, y = xy
            makedir(dir, exists_ok=True)
            x.save(path.join(dir, "x.png"))
            y.save(path.join(dir, "y.png"))

        def save(p, x: Image):
            makedir(path.dirname(p), exists_ok=True)
            x.save(p + ".png")

        xy_paths = list(ptree(self.settings.paths.data.x, self.settings.paths.data.y).pairs())
        cache_dir = makedir(self.settings.paths.cache_dir, hex_of_hash(self.settings.preprocess), exists_ok=True)
        pos = PreProcess(xy_paths, load_pair) \
            .map_k(lambda xy: path.basename(xy[0])) \
            .map_v(lambda _, xy: split_pair(*xy)) \
            .filter_l(lambda _ks, xy: has_foreground(xy[1])) \
            .to_cache(path.join(cache_dir, "pos"), save_pair)

        n_paths = flatten(map(lambda x: path.join(d, x), files) for d, _, files in os.walk(self.settings.paths.data.n))
        neg = PreProcess(n_paths, lambda p: Image.open(p).convert("RGB")) \
            .map_k(lambda n: path.basename(n)) \
            .map_v(lambda _, n: split_single(n)) \
            .to_cache(path.join(cache_dir, "neg"), save)

        return pos, neg

    def get_data_loader(self) -> DataLoader:
        settings = self.settings
        pos, neg = self.get_cache_data()
        pos = pos.build_cache().list_examples()
        neg = neg.build_cache().list_examples()

        class TrainData(Dataset, ABC):
            def __getitem__(self, index):
                self.random_color_jitter = random_color_jitter_fn(
                    brightness=settings.augment.jitter_brightness,
                    contrast=settings.augment.jitter_contrast,
                    saturation=settings.augment.jitter_saturation,
                    hue=settings.augment.jitter_hue,
                )
                self.random_mirror_and_quarter_rot = random_mirror_and_quarter_rot_fn()
                self.pad_tensor = ZeroPad2d(padding=settings.preprocess.image_padding)

        class PosData(TrainData):
            def __getitem__(self, index):
                super(PosData, self).__getitem__(index)

                example = pos[index]
                inp_img = Image.open(path.join(example, "x.png")).convert("RGB")
                tgt_img = Image.open(path.join(example, "y.png")).convert("L")

                # very cheap on-the-fly augmentation
                inp_img, tgt_img = list(map(self.random_mirror_and_quarter_rot, [inp_img, tgt_img]))
                inp_img = self.random_color_jitter(inp_img)

                inp_tensor = f.to_tensor(inp_img)
                tgt_tensor = torch.from_numpy((np.array(tgt_img) > (255 / 2)).astype(np.int64))
                inp_tensor, tgt_tensor = both(self.pad_tensor)((inp_tensor, tgt_tensor))
                assert inp_tensor.dtype == torch.float32
                assert tgt_tensor.dtype == torch.int64

                # from utilities.plotter import img_show
                # img_show(inp_tensor, tgt_tensor, wait=True)
                return inp_tensor, tgt_tensor

            def __len__(self):
                return len(pos)

        class NegData(TrainData):
            def __getitem__(self, index):
                super(NegData, self).__getitem__(index)

                example: str = neg[index]
                inp_img = Image.open(example + ".png").convert("RGB")

                # very cheap on-the-fly augmentation
                inp_img = self.random_mirror_and_quarter_rot(inp_img)
                inp_img = self.random_color_jitter(inp_img)

                inp_tensor = f.to_tensor(inp_img)
                tgt_tensor = torch.zeros(*inp_tensor.shape[-2:], dtype=torch.int64)
                inp_tensor, tgt_tensor = both(self.pad_tensor)((inp_tensor, tgt_tensor))
                assert inp_tensor.dtype == torch.float32
                assert tgt_tensor.dtype == torch.int64

                # from utilities.plotter import img_show
                # img_show(inp_tensor, tgt_tensor, wait=True)
                return inp_tensor, tgt_tensor

            def __len__(self):
                return len(neg)

        train_data = ConcatDataset((PosData(), NegData()))
        return DataLoader(train_data, batch_size=1,
                          shuffle=settings.data_loader.shuffle,
                          num_workers=settings.data_loader.num_workers)

    def run(self) -> None:
        settings = self.settings
        device = self.device

        self.model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=None)
        softmax = torch.nn.Softmax2d()

        dataloader = self.get_data_loader()
        optimizer = SGD(self.model.parameters(),
                        lr=settings.optimizer_settings.lr,
                        momentum=settings.optimizer_settings.momentum)
        scheduler = ReduceLROnPlateau(optimizer)

        class _TrainLoop(TrainLoop):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.add_default_handlers()
                self.add_handler(Event.TRAIN_STEP_END, self.show_output)
                self.add_handler(Event.TRAIN_STEP_END, self.save_checkpoint)
                self.add_handler(Event.TRAINING_RESUMED, self.load_checkpoint)

            def forward_pass(self, example):
                x, _ = example
                return self.model(x.to(device))

            def compute_loss(self, example, yhat):
                _, y = example
                yhat = align_target(y, yhat)
                return loss_fn(yhat, y.to(device))

            def show_output(self, example, yhat, **_kwargs):
                time_training_running = self.state.timer.duration
                delta = time_training_running - self.state.with_default(float("-inf")).time_when_last_shown
                if delta > hms("5s"):
                    self.state.time_when_last_shown = time_training_running

                    print("({t}) example: {nex} of {eps} in epoch {nep} ({nex_t} total)"
                          " ; loss = {loss:.3f} mean epoch loss = {mel:.3f}"
                          " ; {speed:.1f} examples/sec".format(
                            t="{:02d}:{:02d}:{:02d}".format(*time_hms(time_training_running)),
                            loss=self.state.loss,
                            mel=self.state.mean_epoch_loss,
                            speed=self.state.n_examples/time_training_running,
                            nex=self.state.n_examples_current_epoch,
                            nex_t=self.state.n_examples,
                            eps=len(self.dataloader),
                            nep=self.state.n_epochs)
                          )
                    x, y = example
                    segmentation = softmax(yhat)[0, 1].detach()
                    img_show(x, y, segmentation)

            def save_checkpoint(self, **_kwargs):
                time_training_running = self.state.timer.duration
                delta = time_training_running - self.state.with_default(0).time_when_last_saved
                if delta > hms("15s"):
                    self.state.time_when_last_saved = time_training_running

                    cp_path_0 = path.join(settings.paths.checkpoint.dir, "example_{:06d}".format(self.state.n_examples))
                    cp_path_1 = path.join(settings.paths.checkpoint.dir, "last")
                    print("saving to checkpoint: " + cp_path_0)
                    self.save(cp_path_0)
                    self.save(cp_path_1)

            def load_checkpoint(self, resume_from, **_kwargs):
                cp_path = path.join(settings.paths.checkpoint.dir, resume_from)
                print("resuming from checkpoint: " + cp_path)
                self.load(cp_path)

        tl = _TrainLoop(self.model, dataloader, optimizer, scheduler)
        print_param_overview(self.model, optimizer, width=120)
        tl.train(self.settings.paths.checkpoint.resume_from)

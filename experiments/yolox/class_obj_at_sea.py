"""YOLOX Exp class for the DCS sea-domain detection model.

Overrides the upstream YOLOX defaults to match the project's locked decisions:

- ``input_size = (640, 640)``, ``test_size = (640, 640)``
- ``multiscale_range = 0`` — multiscale training disabled
- ``num_classes = 3`` (ships, helicopters, airplanes; canonical contract in
  ``packages/common/classes.py``) — the actual value is injected at runtime by
  ``apps.yolox_training.exp_factory.apply_exp_overrides`` from dataset metadata
- ``no_aug_epochs`` configurable (default 15) — last N epochs disable mosaic+mixup
- ``save_history_ckpt = False`` — per-epoch saves handled by ``CustomTrainer``
- ``get_data_loader`` / ``get_eval_loader`` use the pipeline layout
  (``images/{train,val,test}`` instead of YOLOX's stock ``train2017``).

Loading this module is a hard yolox dependency. If yolox is not installed,
import will fail with a clear error.
"""
from __future__ import annotations

import os

# Hard dependency on yolox — only fails on import time if yolox is missing.
from yolox.exp import Exp as YoloxBaseExp

_DEFAULT_NUM_CLASSES = 3  # ships, helicopters, airplanes (see packages/common/classes.py)
_DEFAULT_TRAIN_IMG_SUBDIR = "images/train"
_DEFAULT_VAL_IMG_SUBDIR   = "images/val"
_DEFAULT_TEST_IMG_SUBDIR  = "images/test"


class Exp(YoloxBaseExp):
    def __init__(self) -> None:
        super().__init__()

        # Model size (yolox_s defaults) — can be overridden by apply_exp_overrides
        # if we later want a larger/smaller backbone.
        self.depth = 0.33
        self.width = 0.50

        # ---------- locked decisions ----------
        self.num_classes      = _DEFAULT_NUM_CLASSES
        self.input_size       = (640, 640)
        self.test_size        = (640, 640)
        self.multiscale_range = 0          # ← TARGET §15

        # Training schedule — actual values come from pipeline.yaml at runtime.
        self.max_epoch     = 100
        self.no_aug_epochs = 15
        self.eval_interval = 10

        # Checkpoints — let CustomTrainer take over per-epoch saves.
        self.save_history_ckpt = False

        # Data — paths get filled in by apply_exp_overrides:
        # data_dir = .../enriched_dataset, train_ann = "instances_train.json", etc.
        self.train_ann = "instances_train.json"
        self.val_ann   = "instances_val.json"
        self.test_ann  = "instances_test.json"

        # Image subdirs relative to data_dir. apply_exp_overrides may rewrite
        # these for non-standard layouts.
        self.pipeline_train_image_subdir = _DEFAULT_TRAIN_IMG_SUBDIR
        self.pipeline_val_image_subdir   = _DEFAULT_VAL_IMG_SUBDIR
        self.pipeline_test_image_subdir  = _DEFAULT_TEST_IMG_SUBDIR

        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    # ------------------------------------------------------------------
    # Data loaders — must reroute to our layout images/{train,val,test}/.
    # The upstream methods hardcode "train2017" / "val2017"; we copy the
    # surrounding logic and only swap the ``name`` argument of COCODataset.
    # ------------------------------------------------------------------

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        import torch.distributed as dist
        from yolox.data import (
            COCODataset,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            TrainTransform,
            YoloBatchSampler,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name=self.pipeline_train_image_subdir,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                ),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset),
                                   seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )
        return DataLoader(
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.data_num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_reset_seed,
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        import torch
        import torch.distributed as dist
        from yolox.data import COCODataset, ValTransform

        json_file = self.test_ann if testdev else self.val_ann
        name = (
            self.pipeline_test_image_subdir if testdev
            else self.pipeline_val_image_subdir
        )

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=json_file,
            name=name,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False,
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        return torch.utils.data.DataLoader(
            valdataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=batch_size,
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator
        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed, testdev, legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )

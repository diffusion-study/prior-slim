import json
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

from coca_pytorch import CoCa
from open_clip import list_pretrained
from pydantic import BaseModel, model_validator
from x_clip import CLIP as XCLIP

from dalle2_pytorch.dalle2_pytorch import (CoCaAdapter, DiffusionPrior,
                                           DiffusionPriorNetwork,
                                           OpenAIClipAdapter, OpenClipAdapter,
                                           XClipAdapter)
from dalle2_pytorch.trackers import (Tracker, create_loader, create_logger,
                                     create_saver)

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


InnerType = TypeVar("InnerType")
ListOrTuple = Union[List[InnerType], Tuple[InnerType]]
SingularOrIterable = Union[InnerType, ListOrTuple[InnerType]]

# general pydantic classes


class TrainSplitConfig(BaseModel):
    train: float = 0.75
    val: float = 0.15
    test: float = 0.1

    @model_validator(mode="after")
    def validate_all(self, m):
        actual_sum = sum([*dict(self).values()])
        if actual_sum != 1.0:
            raise ValueError(
                f"{dict(self).keys()} must sum to 1.0. Found: {actual_sum}"
            )
        return self


class TrackerLogConfig(BaseModel):
    log_type: str = "console"
    resume: bool = (
        False  # For logs that are saved to unique locations, resume a previous run
    )
    auto_resume: bool = (
        False  # If the process crashes and restarts, resume from the run that crashed
    )
    verbose: bool = False

    class Config:
        # Each individual log type has it's own arguments that will be passed through the config
        extra = "allow"

    def create(self, data_path: str):
        kwargs = self.dict()
        return create_logger(self.log_type, data_path, **kwargs)


class TrackerLoadConfig(BaseModel):
    load_from: Optional[str] = None
    only_auto_resume: bool = (
        False  # Only attempt to load if the logger is auto-resuming
    )

    class Config:
        extra = "allow"

    def create(self, data_path: str):
        kwargs = self.dict()
        if self.load_from is None:
            return None
        return create_loader(self.load_from, data_path, **kwargs)


class TrackerSaveConfig(BaseModel):
    save_to: str = "local"
    save_all: bool = False
    save_latest: bool = True
    save_best: bool = True

    class Config:
        extra = "allow"

    def create(self, data_path: str):
        kwargs = self.dict()
        return create_saver(self.save_to, data_path, **kwargs)


class TrackerConfig(BaseModel):
    data_path: str = ".tracker_data"
    overwrite_data_path: bool = False
    log: TrackerLogConfig
    load: Optional[TrackerLoadConfig] = None
    save: Union[List[TrackerSaveConfig], TrackerSaveConfig]

    def create(
        self, full_config: BaseModel, extra_config: dict, dummy_mode: bool = False
    ) -> Tracker:
        tracker = Tracker(
            self.data_path,
            dummy_mode=dummy_mode,
            overwrite_data_path=self.overwrite_data_path,
        )
        # Add the logger
        tracker.add_logger(self.log.create(self.data_path))
        # Add the loader
        if self.load is not None:
            tracker.add_loader(self.load.create(self.data_path))
        # Add the saver or savers
        if isinstance(self.save, list):
            for save_config in self.save:
                tracker.add_saver(save_config.create(self.data_path))
        else:
            tracker.add_saver(self.save.create(self.data_path))
        # Initialize all the components and verify that all data is valid
        tracker.init(full_config, extra_config)
        return tracker


# diffusion prior pydantic classes


class AdapterConfig(BaseModel):
    make: str = "openai"
    model: str = "ViT-L/14"
    base_model_kwargs: Optional[Dict[str, Any]] = None

    def create(self):
        if self.make == "openai":
            return OpenAIClipAdapter(self.model)
        elif self.make == "open_clip":
            pretrained = dict(list_pretrained())
            checkpoint = pretrained[self.model]
            return OpenClipAdapter(name=self.model, pretrained=checkpoint)
        elif self.make == "x-clip":
            return XClipAdapter(XCLIP(**self.base_model_kwargs))
        elif self.make == "coca":
            return CoCaAdapter(CoCa(**self.base_model_kwargs))
        else:
            raise AttributeError("No adapter with that name is available.")


class DiffusionPriorNetworkConfig(BaseModel):
    dim: int
    depth: int
    max_text_len: Optional[int] = None
    num_timesteps: Optional[int] = None
    num_time_embeds: int = 1
    num_image_embeds: int = 1
    num_text_embeds: int = 1
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4
    norm_in: bool = False
    norm_out: bool = True
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    final_proj: bool = True
    normformer: bool = False
    rotary_emb: bool = True

    class Config:
        extra = "allow"

    def create(self):
        kwargs = self.dict()
        return DiffusionPriorNetwork(**kwargs)


class DiffusionPriorConfig(BaseModel):
    clip: Optional[AdapterConfig] = None
    net: DiffusionPriorNetworkConfig
    image_embed_dim: int
    image_size: int
    image_channels: int = 3
    timesteps: int = 1000
    sample_timesteps: Optional[int] = None
    cond_drop_prob: float = 0.0
    loss_type: str = "l2"
    predict_x_start: bool = True
    beta_schedule: str = "cosine"
    condition_on_text_encodings: bool = True

    class Config:
        extra = "allow"

    def create(self):
        kwargs = self.dict()

        has_clip = exists(kwargs.pop("clip"))
        kwargs.pop("net")

        clip = None
        if has_clip:
            clip = self.clip.create()

        diffusion_prior_network = self.net.create()
        return DiffusionPrior(net=diffusion_prior_network, clip=clip, **kwargs)


class DiffusionPriorTrainConfig(BaseModel):
    epochs: int = 1
    lr: float = 1.1e-4
    wd: float = 6.02e-2
    max_grad_norm: float = 0.5
    use_ema: bool = True
    ema_beta: float = 0.99
    amp: bool = False
    warmup_steps: Optional[int] = None  # number of warmup steps
    save_every_seconds: int = 3600  # how often to save
    eval_timesteps: List[int] = [64]  # which sampling timesteps to evaluate with
    best_validation_loss: float = 1e9  # the current best valudation loss observed
    current_epoch: int = 0  # the current epoch
    num_samples_seen: int = 0  # the current number of samples seen
    random_seed: int = 0  # manual seed for torch


class DiffusionPriorDataConfig(BaseModel):
    image_url: str  # path to embeddings folder
    meta_url: str  # path to metadata (captions) for images
    splits: TrainSplitConfig  # define train, validation, test splits for your dataset
    batch_size: int  # per-gpu batch size used to train the model
    num_data_points: int = 25e7  # total number of datapoints to train on
    eval_every_seconds: int = 3600  # validation statistics will be performed this often


class TrainDiffusionPriorConfig(BaseModel):
    prior: DiffusionPriorConfig
    data: DiffusionPriorDataConfig
    train: DiffusionPriorTrainConfig
    tracker: TrackerConfig

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)

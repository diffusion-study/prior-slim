## Dataloaders
In order to make loading data simple and efficient, we include some general dataloaders that can be used to train portions of the network.


### Diffusion Prior: Prior Embedding Dataset
When training the prior it is much more efficient to work with pre-computed embeddings. The `PriorEmbeddingDataset` class enables you to leverage the same script (with minimal modification) for both embedding-only and text-conditioned prior training. This saves you from having to worry about a lot of the boilerplate code.

To utilize the `PriorEmbeddingDataset`, all you need to do is make a single call to `get_reader()` which will create `EmbeddingReader` object(s) for you. Afterwards, you can utilize `make_splits()` to cleanly create DataLoader objects from for your training run.

If you are training in a distributed manner, `make_splits()` accepts `rank` and `world_size` arguments to properly distribute to each process. The defaults for these values are `rank=0` and `world_size=1`, so single-process training can safely ignore these parameters.

Usage:
```python
from dalle2_pytorch.dataloaders import get_reader, make_splits

# grab embeddings from some specified location
IMG_URL = "data/img_emb/"
META_URL = "data/meta/"

reader = get_reader(text_conditioned=True, img_url=IMG_URL, meta_url=META_URL)

# some config for training
TRAIN_ARGS = {
    "world_size": 3,
    "text_conditioned": True,
    "start": 0,
    "num_data_points": 10000,
    "batch_size": 2,
    "train_split": 0.5,
    "eval_split": 0.25,
    "image_reader": reader,
}

# specifying a rank will handle allocation internally
rank0_train, rank0_eval, rank0_test = make_splits(rank=0, **TRAIN_ARGS)
rank1_train, rank1_eval, rank1_test = make_splits(rank=1, **TRAIN_ARGS)
rank2_train, rank2_eval, rank2_test = make_splits(rank=2, **TRAIN_ARGS)
```

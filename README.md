# Disentanglement Datasets

A lightweight package of `torchvision`-style PyTorch datasets with a focus on flexibility.

## Installation

    $ pip install pytorch-disentanglement-datasets

## Usage

Each dataset returns a dictionary containing at least an `input` key:

```python
>>> from disentanglement_datasets import DSprites
>>> dataset = DSprites(root="./data", download=True)
>>> dataset[0]
{'input': tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]], dtype=torch.uint8),
 'latent': tensor([1.0000, 1.0000, 0.5000, 0.0000, 0.0000, 0.0000], dtype=torch.float64)}
 ```

This dictionary can be transformed. For example, for self-supervised learning with augmentations you might do something like:

```python
>>> def my_transform(item):
...   view1 = my_random_augmentation(item["input"])
...   view2 = my_random_augmentation(item["input"])
...   return view1, view2
...
>>> dataset = DSprites(root="./data", download=True, transform=my_transform)
>>> dataset[0]
(tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]], dtype=torch.uint8), tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]], dtype=torch.uint8))
```

## Datasets and their original sources

* [DSprites](https://github.com/deepmind/dsprites-dataset)
* [MPI3D-Toy](https://github.com/rr-learning/disentanglement_dataset)

## Attribution

If this code was useful to you, please cite the original dataset (links above) and this codebase.

### BibTeX

TODO

## Development

Pull requests are welcome.

## See also

* [`disentanglement-pytorch`](https://github.com/amir-abdi/disentanglement-pytorch): Variational autoencoder models for disentanglement, created as a contribution to the [*Disentanglement Challenge of NeurIPS 2019*](https://aicrowd.com/challenges/neurips-2019-disentanglement-challenge), along with more datasets with a different interface.
* [`disentanglement_lib`](https://github.com/google-research/disentanglement_lib): VAE models, metrics, and data for disentanglement, implemented in TensorFlow, created as a contribution to the [*Disentanglement Challenge of NeurIPS 2019*](https://aicrowd.com/challenges/neurips-2019-disentanglement-challenge).

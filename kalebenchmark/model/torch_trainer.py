"""Domain-neutral training for the ``predict`` stage

Running a PyTorch Lightning module over a dataset is the same work in every discipline: build a
loader, fit, then run the network over the evaluation partition. What differs is the module that
defines the loss and metrics, the collate function that batches the records, and the network
itself, so :class:`TorchRegressor` takes all three from its caller and implements none of them.

Released PyKale provides no such runner: :class:`kale.pipeline.base_nn_trainer.BaseNNTrainer` is
a Lightning module, and every ``examples/*/main.py`` constructs ``pl.Trainer`` inline for itself.

TODO(pykale): upstream this once a general trainer entry point exists there.
"""

import logging
from typing import Any, Callable, Optional

import numpy as np


class TorchRegressor:
    """Fit a Lightning module on a dataset, then predict one value per record.

    A subclass overrides :meth:`build_module` and :meth:`collate` when those depend on a
    discipline; see :mod:`kalebenchmark.model.materials.crystal_gnn`.

    Args:
        module (optional): A Lightning module, or a callable taking one record and returning one.
        collate_fn (callable, optional): Batches records, as ``DataLoader`` expects. Defaults to
            None, which leaves the loader's own default in place.
        max_epochs (int, optional): Training epochs. Defaults to 30.
        batch_size (int, optional): Batch size. Defaults to 64.
        accelerator (str, optional): Lightning accelerator. Defaults to "cpu".
        devices (int, optional): Number of devices. Defaults to 1.
        num_workers (int, optional): Data loader workers. Defaults to 0.
    """

    def __init__(
        self,
        module: Optional[Any] = None,
        collate_fn: Optional[Callable] = None,
        max_epochs: int = 30,
        batch_size: int = 64,
        accelerator: str = "cpu",
        devices: int = 1,
        num_workers: int = 0,
    ) -> None:
        self.module = module
        self.collate_fn = collate_fn
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.devices = devices
        self.num_workers = num_workers
        self.fitted: Optional[Any] = None

    def collate(self) -> Optional[Callable]:
        """Return the collate function, or None for the loader's default.

        Override this to import a discipline's collate function only when a run needs it.
        """
        return self.collate_fn

    def build_module(self, sample: Any) -> Any:
        """Return the Lightning module to fit, sized from one record if needed.

        Args:
            sample: The first record, for modules whose widths come from the data.

        Returns:
            The Lightning module.

        Raises:
            ValueError: If no module was given.
        """
        if self.module is None:
            raise ValueError(f"{type(self).__name__} needs a module, or a build_module() override.")
        if hasattr(self.module, "training_step"):
            return self.module
        return self.module(sample)

    def loader(self, data: Any, shuffle: bool = False) -> Any:
        """Build the data loader for one partition."""
        from torch.utils.data import DataLoader

        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate(),
        )

    def fit(self, data: Any, targets: Optional[Any] = None) -> "TorchRegressor":
        """Train on a dataset whose records carry their own targets.

        Args:
            data: The training partition.
            targets (optional): Ignored; accepted for protocol compatibility.

        Returns:
            TorchRegressor: This object.
        """
        import pytorch_lightning as pl

        self.fitted = self.build_module(data[0])
        pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        ).fit(self.fitted, self.loader(data, shuffle=True))
        return self

    def predict(self, data: Any) -> np.ndarray:
        """Predict one value per record.

        Args:
            data: The partition to predict.

        Returns:
            numpy.ndarray: One prediction per record.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        import torch

        if self.fitted is None:
            error_msg = f"{type(self).__name__}.predict called before fit."
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        self.fitted.eval()
        outputs = []
        with torch.no_grad():
            for batch in self.loader(data):
                output = self.fitted(batch)
                outputs.append(output[0] if isinstance(output, (tuple, list)) else output)
        return np.concatenate([output.cpu().numpy().ravel() for output in outputs])

"""Crystal graph networks for the ``predict`` stage

Running the training is domain neutral and lives in
:class:`examples.realmat_bag.torch_trainer.TorchRegressor`. This module only injects the three
things that are specific to crystals, all of which the RealMat-BaG reference already implements:
its ``MaterialsTrainer`` (the loss, the metrics and the optimisers), its collate function, and
its networks. Nothing here trains, batches or defines a loss.

Construction is per architecture, so a network from anywhere else needs no code here: pass a
``torch.nn.Module``, or a callable sized from one crystal.

Classical estimators need no module here at all. Their shortcuts resolve straight to
:class:`sklearn.svm.SVR`, :class:`sklearn.ensemble.RandomForestRegressor` and
:class:`sklearn.linear_model.LinearRegression`, which already provide ``fit``/``predict``.
"""

from typing import Any, Callable, Dict, Optional, Union

from examples.realmat_bag.realmat_bag import PathLike, require_realmat_bag
from examples.realmat_bag.torch_trainer import TorchRegressor


def build_cgcnn(sample: Any, atom_fea_len: int = 64, n_conv: int = 3, h_fea_len: int = 128, n_h: int = 1) -> Any:
    """Build a CGCNN network sized from one crystal.

    Args:
        sample: One crystal, giving the input feature widths.
        atom_fea_len (int, optional): Hidden atom feature width. Defaults to 64.
        n_conv (int, optional): Number of convolution layers. Defaults to 3.
        h_fea_len (int, optional): Width of the dense head. Defaults to 128.
        n_h (int, optional): Number of dense layers. Defaults to 1.

    Returns:
        torch.nn.Module: The network.
    """
    from realmat_bag.pipeline.models.cgcnn.CGCNN import CrystalGraphConvNet

    return CrystalGraphConvNet(
        sample.atom_fea.shape[-1], sample.nbr_fea.shape[-1], atom_fea_len, n_conv, h_fea_len, n_h
    )


def build_leftnet(
    sample: Any,
    encoding: str = "prop",
    hidden_channels: int = 128,
    num_layers: int = 4,
    num_radial: int = 96,
    cutoff: float = 8.0,
    **kwargs: Any,
) -> Any:
    """Build a LEFTNet network sized from one crystal.

    Args:
        sample: One crystal, giving the input feature widths.
        encoding (str, optional): "prop" for atom properties, "z" for atomic number.
        hidden_channels (int, optional): Hidden width. Defaults to 128.
        num_layers (int, optional): Message-passing layers. Defaults to 4.
        num_radial (int, optional): Radial basis size. Defaults to 96.
        cutoff (float, optional): Neighbour cutoff, in the graphs' own units. Defaults to 8.0.
        **kwargs: Forwarded to the network.

    Returns:
        torch.nn.Module: The network.
    """
    from realmat_bag.pipeline.models.leftnet.leftnet import LEFTNet

    return LEFTNet(
        bond_feat_dim=sample.nbr_fea.shape[-1],
        num_targets=1,
        encoding=encoding,
        prop_input_dim=sample.atom_fea.shape[-1],
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_radial=num_radial,
        cutoff=cutoff,
        **kwargs,
    )


ARCHITECTURES: Dict[str, Callable[..., Any]] = {"cgcnn": build_cgcnn, "leftnet": build_leftnet}


class CrystalGraphRegressor(TorchRegressor):
    """A crystal network trained by the RealMat-BaG Lightning module.

    Args:
        model (str, torch.nn.Module or callable, optional): A name in :data:`ARCHITECTURES`, a
            ready network, or a callable taking one crystal and returning a network. Defaults to
            "cgcnn".
        model_params (dict, optional): Passed to the architecture builder.
        optimizer (dict, optional): Optimiser spec for the Lightning module. Defaults to Adam.
        init_lr (float, optional): Initial learning rate. Defaults to 0.001.
        layer_freeze (str, optional): Freezing mode of the Lightning module. Defaults to "none".
        root (str or Path, optional): Checkout root. Defaults to ``REALMAT_BAG_ROOT`` or
            ``./bandgap-benchmark``.
        **kwargs: Passed to :class:`~examples.realmat_bag.torch_trainer.TorchRegressor`, which
            documents ``max_epochs``, ``batch_size``, ``accelerator``, ``devices`` and
            ``num_workers``.

    Examples:
        >>> from examples.realmat_bag.crystal_gnn import CrystalGraphRegressor
        >>> CrystalGraphRegressor(model="leftnet", max_epochs=50)  # doctest: +SKIP
    """

    def __init__(
        self,
        model: Union[str, Any] = "cgcnn",
        model_params: Optional[Dict[str, Any]] = None,
        optimizer: Optional[Dict[str, Any]] = None,
        init_lr: float = 0.001,
        layer_freeze: str = "none",
        root: Optional[PathLike] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.model_params = model_params or {}
        self.optimizer = optimizer or {"type": "Adam", "optim_params": {}}
        self.init_lr = init_lr
        self.layer_freeze = layer_freeze
        self.root = root

    def collate(self) -> Callable:
        """Return the reference batching of variable-size crystals into one flat graph."""
        # TODO(pykale): use kale.loaddata.materials_datasets.CIFData.collate_fn once released.
        require_realmat_bag(self.root)
        from realmat_bag.loaddata.collate import collate_crystal_batch

        return collate_crystal_batch

    def build_network(self, sample: Any) -> Any:
        """Return the network, building it from ``sample`` unless one was given ready."""
        if isinstance(self.model, str):
            if self.model not in ARCHITECTURES:
                raise KeyError(f"Unknown architecture {self.model!r}. Known: {sorted(ARCHITECTURES)}")
            return ARCHITECTURES[self.model](sample, **self.model_params)
        if callable(self.model) and not hasattr(self.model, "forward"):
            return self.model(sample, **self.model_params)
        return self.model

    def build_module(self, sample: Any) -> Any:
        """Wrap the network in the reference Lightning module, which owns the loss and metrics."""
        # TODO(pykale): use kale.pipeline.base_nn_trainer once the materials work is released.
        require_realmat_bag(self.root)
        from realmat_bag.pipeline.trainer import MaterialsTrainer

        return MaterialsTrainer(
            model=self.build_network(sample),
            optimizer=self.optimizer,
            max_epochs=self.max_epochs,
            layer_freeze=self.layer_freeze,
            init_lr=self.init_lr,
        )

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np

from nflows import distributions, flows, transforms, utils
import nflows.nn.nets as nn_

from pathlib import Path

from nflows.transforms.base import Transform
from nflows.transforms.autoregressive import AutoregressiveTransform
from nflows.transforms import made as made_module

from modded_spline import (
    unconstrained_rational_quadratic_spline,
    rational_quadratic_spline,
)
import modded_spline
from nflows.utils import torchutils

# from nflows.transforms import splines
from torch.nn.functional import softplus

from modded_coupling import PiecewiseCouplingTransformRQS, CouplingTransformMAF
from modded_base_flow import FlowM
from nflows.flows.base import Flow


class AffineCouplingTransform(CouplingTransformMAF):
    """An affine coupling layer that scales and shifts part of the variables.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.

    The user should supply `scale_activation`, the final activation function in the neural network producing the scale tensor.
    Two options are predefined in the class.
    `DEFAULT_SCALE_ACTIVATION` preserves backwards compatibility but only produces scales <= 1.001.
    `GENERAL_SCALE_ACTIVATION` produces scales <= 3, which is more useful in general applications.
    """

    DEFAULT_SCALE_ACTIVATION = lambda x: torch.sigmoid(x + 2) + 1e-3
    GENERAL_SCALE_ACTIVATION = lambda x: (softplus(x) + 1e-3).clamp(0, 30)

    def __init__(
        self,
        mask,
        transform_net_create_fn,
        unconditional_transform=None,
        scale_activation="softplus",
        init_identity=True,
        shift_clamp=[-50, 50],
        scale_clamp=[0, 50],
    ):
        self.scale_clamp = scale_clamp
        self.shift_clamp = shift_clamp
        if scale_activation == "softplus":
            self.scale_activation = lambda x: (softplus(x) + 1e-3).clamp(
                self.scale_clamp[0], self.scale_clamp[1]
            )
        self.init_identity = init_identity
        super().__init__(mask, transform_net_create_fn, unconditional_transform)

    def _transform_dim_multiplier(self):
        return 2

    def _scale_and_shift(self, transform_params):
        unconstrained_scale = transform_params[:, self.num_transform_features :, ...]
        shift = transform_params[:, : self.num_transform_features, ...].clamp(
            self.shift_clamp[0], self.shift_clamp[1]
        )
        if self.init_identity:
            shift = shift - 0.5414
        scale = self.scale_activation(unconstrained_scale)
        return scale, shift

    def _coupling_transform_forward(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = inputs * scale + shift
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _coupling_transform_inverse(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet


class MLP(nn.Module):
    def __init__(
        self,
        in_shape: int,
        out_shape: int,
        context_features: int,
        hidden_sizes: list,
        activation=F.relu,
        activate_output: bool = False,
        batch_norm: bool = False,
        dropout_probability: float = 0.0,
    ):
        super().__init__()

        layers = []
        prev_size = in_shape + context_features if context_features else in_shape

        # Initialize hidden layers
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(p=dropout_probability))
            layers.append(activation)
            prev_size = size

        # Final layer
        layers.append(nn.Linear(prev_size, out_shape))
        if activate_output:
            layers.append(activation)

        self.network = nn.Sequential(*layers)

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor = None
    ) -> torch.Tensor:
        x = torch.cat((inputs, context), dim=1) if context else inputs
        return self.network(x)


class EmbedATT(nn.Module):
    def __init__(
        self,
        in_shape,
        embed_shape,
        out_shape,
        context_features,
        hidden_sizes,
        activation=F.relu,
        activate_output=False,
        layer_norm=False,
        dropout_probability=0.0,
        num_heads=5,
    ):
        super().__init__()
        self._in_shape = in_shape
        self._embed_shape = embed_shape
        self._out_shape = out_shape
        self._context_features = context_features
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._activate_output = activate_output
        self._layer_norm = layer_norm
        self.dropout = nn.Dropout(p=dropout_probability)
        self.num_heads = num_heads

        if len(hidden_sizes) == 0:
            raise ValueError("List of hidden sizes can't be empty.")

        self.input_size = (
            in_shape if context_features is None else in_shape + context_features
        )
        self.embedding = nn.Linear(self.input_size, self.input_size * embed_shape)

        if layer_norm:
            self.layer_norm_layers = nn.ModuleList(
                [torch.nn.LayerNorm(sizes) for sizes in hidden_sizes]
            )

        self.attention = torch.nn.MultiheadAttention(
            embed_dim=embed_shape, num_heads=self.num_heads, batch_first=True
        )

        self._hidden_layers = nn.ModuleList(
            [
                nn.Linear(self._embed_shape * self.input_size, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 128),
            ]
        )
        self.final_layer = nn.Linear(128, np.prod(out_shape))

    def forward(self, inputs, context=None):
        if context is not None:
            combined_input = torch.cat((inputs, context), dim=1)
        else:
            combined_input = inputs

        embedded = self.embedding(combined_input).view(
            -1, self.input_size, self._embed_shape
        )
        attn_output, _ = self.attention(
            embedded, embedded, embedded, need_weights=False
        )

        # Simple aggregation using mean
        attn_aggregated = torch.mean(attn_output, dim=1)

        outputs = attn_aggregated

        for i, hidden_layer in enumerate(self._hidden_layers):
            outputs = hidden_layer(outputs)
            if self._layer_norm:
                outputs = self.layer_norm_layers[i](outputs)
            outputs = self._activation(outputs)
            outputs = self.dropout(outputs)

        outputs = self.final_layer(outputs)

        if self._activate_output:
            outputs = self._activation(outputs)

        return outputs


class MaskedAffineAutoregressiveTransformM(AutoregressiveTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        init_identity=True,
        affine_type="sigmoid",
        shift_clamp=[-50, 50],
        scale_clamp=[0, 50],
    ):
        self.features = features
        made = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        self._epsilon = 5e-2
        self.shift_clamp = shift_clamp
        self.scale_clamp = scale_clamp
        self.init_identity = init_identity
        self.affine_type = affine_type
        if init_identity:
            torch.nn.init.constant_(made.final_layer.weight, 0.0)
            if self.affine_type == "softplus":
                torch.nn.init.constant_(
                    made.final_layer.bias,
                    0.5414,  # the value k to get softplus(k) = 1.0
                )
            elif self.affine_type == "sigmoid":
                torch.nn.init.constant_(
                    made.final_layer.bias,
                    -7.906,  # the value k to get sigmoid(k+1) = 1.0
                )
            elif self.affine_type == "atan":
                torch.nn.init.constant_(
                    made.final_layer.bias, 1  # the value k to get atan(k) = 1.0
                )

        super(MaskedAffineAutoregressiveTransformM, self).__init__(made)

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        shift = shift.clamp(self.shift_clamp[0], self.shift_clamp[1])
        if self.affine_type == "sigmoid":
            scale = 1000 * torch.sigmoid(unconstrained_scale + 1.0) + self._epsilon
        elif self.affine_type == "softplus":
            scale = ((F.softplus(unconstrained_scale)) + self._epsilon).clamp(
                self.scale_clamp[0], self.scale_clamp[1]
            )
        elif self.affine_type == "atan":
            scale = (1000 * torch.atan(unconstrained_scale / 1000)).clamp(
                self.scale_clamp[0], self.scale_clamp[1]
            )
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift

        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        shift = shift.clamp(self.shift_clamp[0], self.shift_clamp[1])
        if self.affine_type == "sigmoid":
            scale = 1000 * torch.sigmoid(unconstrained_scale + 1.0) + self._epsilon
        elif self.affine_type == "softplus":
            scale = ((F.softplus(unconstrained_scale)) + self._epsilon).clamp(
                self.scale_clamp[0], self.scale_clamp[1]
            )
        elif self.affine_type == "atan":
            scale = (1000 * torch.atan(unconstrained_scale / 1000)).clamp(
                self.scale_clamp[0], self.scale_clamp[1]
            )
        log_scale = torch.log(scale)
        # print(scale, shift)
        outputs = (inputs - shift) / scale

        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, autoregressive_params):
        # split_idx = autoregressive_params.size(1) // 2
        # unconstrained_scale = autoregressive_params[..., :split_idx]
        # shift = autoregressive_params[..., split_idx:]
        # return unconstrained_scale, shift
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        unconstrained_scale = autoregressive_params[..., 0]
        shift = autoregressive_params[..., 1]
        if self.init_identity:
            if self.affine_type == "sigmoid":
                shift = shift + 7.906
            elif self.affine_type == "softplus":
                shift = shift - 0.5414
            elif self.affine_type == "atan":
                shift = shift - 1
        # print(unconstrained_scale, shift)
        return unconstrained_scale, shift


class MaskedPiecewiseRationalQuadraticAutoregressiveTransformM(AutoregressiveTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        init_identity=True,
        min_bin_width=modded_spline.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=modded_spline.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=modded_spline.DEFAULT_MIN_DERIVATIVE,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        autoregressive_net = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        if init_identity:
            torch.nn.init.constant_(autoregressive_net.final_layer.weight, 0.0)
            torch.nn.init.constant_(
                autoregressive_net.final_layer.bias,
                np.log(np.exp(1 - min_derivative) - 1),
            )

        super().__init__(autoregressive_net)

    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs,
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)


class PiecewiseRationalQuadraticCouplingTransformM(PiecewiseCouplingTransformRQS):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        apply_unconditional_transform=False,
        img_shape=None,
        init_identity=True,
        min_bin_width=modded_spline.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=modded_spline.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=modded_spline.DEFAULT_MIN_DERIVATIVE,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = lambda features: PiecewiseRationalQuadraticCDF(
                shape=[features] + (img_shape if img_shape else []),
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            )
        else:
            unconditional_transform = None

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        else:
            return self.num_bins * 3 + 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn(
                "Inputs to the softmax are not scaled down: initialization might be bad."
            )

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs,
        )


def create_random_transform(param_dim):
    """Create the composite linear transform PLU.
    Arguments:
        input_dim {int} -- dimension of the space
    Returns:
        Transform -- nde.Transform object
    """

    return transforms.CompositeTransform(
        [
            transforms.RandomPermutation(features=param_dim),
            # transforms.LULinear(param_dim, identity_init=True),
        ]
    )


def create_mixture_flow_model(input_dim, context_dim, base_kwargs):
    """Build NSF (neural spline flow) model. This uses the nsf module
    available at https://github.com/bayesiains/nsf.
    This models the posterior distribution p(x|y).
    The model consists of
        * a base distribution (StandardNormal, dim(x))
        * a sequence of transforms, each conditioned on y
    Arguments:
        input_dim {int} -- dimensionality of x
        context_dim {int} -- dimensionality of y
        num_flow_steps {int} -- number of sequential transforms
        base_transform_kwargs {dict} -- hyperparameters for transform steps: should put num_transform_blocks=10,
                          activation='elu',
                          batch_norm=True
    Returns:
        Flow -- the model
    """

    distribution = distributions.StandardNormal((input_dim,))
    transform = []
    for _ in range(base_kwargs["maf"]["num_steps"]):

        if base_kwargs["maf"]["activation"] == "relu":
            activationDef = F.relu
        if base_kwargs["maf"]["activation"] == "gelu":
            activationDef = F.gelu
        elif base_kwargs["maf"]["activation"] == "elu":
            activationDef = F.elu
        elif base_kwargs["maf"]["activation"] == "softplus":
            activationDef = F.softplus
        elif base_kwargs["maf"]["activation"] == "leaky_relu":
            activationDef = F.leaky_relu 
        elif base_kwargs["maf"]["activation"] == "silu":
            activationDef = F.silu
        else:
            raise ValueError("Unknown activation")
        
        transform.append(
            MaskedAffineAutoregressiveTransformM(
                features=input_dim,
                use_residual_blocks=base_kwargs["maf"]["use_residual_blocks"],
                num_blocks=base_kwargs["maf"]["num_transform_blocks"],
                hidden_features=base_kwargs["maf"]["hidden_dim"],
                context_features=context_dim,
                dropout_probability=base_kwargs["maf"]["dropout_probability"],
                use_batch_norm=base_kwargs["maf"]["batch_norm"],
                init_identity=base_kwargs["maf"]["init_identity"],
                affine_type=base_kwargs["maf"]["affine_type"],
                shift_clamp=base_kwargs["maf"]["shift_clamp"],
                scale_clamp=base_kwargs["maf"]["scale_clamp"],
            )
        )
        if base_kwargs["permute_type"] != "no-permutation":
            transform.append(create_random_transform(param_dim=input_dim))

    for _ in range(base_kwargs["cmaf"]["num_steps"]):
        if base_kwargs["cmaf"]["activation"] == "relu":
            activation = F.relu
        elif base_kwargs["cmaf"]["activation"] == "gelu":
            activation = F.gelu
        elif base_kwargs["cmaf"]["activation"] == "elu":
            activation = F.elu
        elif base_kwargs["cmaf"]["activation"] == "softplus":
            activation = F.softplus
        elif base_kwargs["cmaf"]["activation"] == "leaky_relu":
            activation = F.leaky_relu
        elif base_kwargs["cmaf"]["activation"] == "silu":
            activation = F.silu
        else:
            raise ValueError("Unknown activation")

        if base_kwargs["cmaf"]["net_type"] == "resnet":
            transform_net_create_fn = lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=base_kwargs["cmaf"]["hidden_dim"],
                context_features=context_dim,
                num_blocks=base_kwargs["cmaf"]["num_transform_blocks"],
                activation=activation,
                dropout_probability=base_kwargs["cmaf"]["dropout_probability"],
                use_batch_norm=base_kwargs["cmaf"]["batch_norm"],
            )
        elif base_kwargs["cmaf"]["net_type"] == "mlp":
            transform_net_create_fn = lambda in_features, out_features: MLP(
                in_features,
                out_features,
                context_features=context_dim,
                hidden_sizes=base_kwargs["cmaf"]["hidden_dim"],
                activation=base_kwargs["cmaf"]["activation"],
                dropout_probability=base_kwargs["cmaf"]["dropout_probability"],
                use_batch_norm=base_kwargs["cmaf"]["batch_norm"],
            )
        else:
            raise ValueError("Unknown net_type")

        transform.append(
            AffineCouplingTransform(
                mask=utils.create_alternating_binary_mask(
                    features=input_dim, even=True
                ),
                transform_net_create_fn=transform_net_create_fn,
                shift_clamp=base_kwargs["cmaf"]["shift_clamp"],
                scale_clamp=base_kwargs["cmaf"]["scale_clamp"],
                init_identity=base_kwargs["cmaf"]["init_identity"],
            )
        )
        if base_kwargs["permute_type"] != "no-permutation":
            transform.append(create_random_transform(param_dim=input_dim))

    for _ in range(base_kwargs["arqs"]["num_steps"]):
        transform.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransformM(
                features=input_dim,
                tails="linear",
                use_residual_blocks=base_kwargs["arqs"]["use_residual_blocks"],
                hidden_features=base_kwargs["arqs"]["hidden_dim"],
                num_blocks=base_kwargs["arqs"]["num_transform_blocks"],
                tail_bound=base_kwargs["arqs"]["tail_bound"],
                num_bins=base_kwargs["arqs"]["num_bins"],
                context_features=context_dim,
                dropout_probability=base_kwargs["arqs"]["dropout_probability"],
                use_batch_norm=base_kwargs["arqs"]["batch_norm"],
                init_identity=base_kwargs["arqs"]["init_identity"],
            )
        )
        if base_kwargs["permute_type"] != "no-permutation":
            transform.append(create_random_transform(param_dim=input_dim))

    for _ in range(base_kwargs["crqs"]["num_steps"]):
        if base_kwargs["crqs"]["activation"] == "relu":
            activation = F.relu
        elif base_kwargs["crqs"]["activation"] == "elu":
            activation = F.elu
        elif base_kwargs["crqs"]["activation"] == "softplus":
            activation = F.softplus
        else:
            raise ValueError("Unknown activation")

        if base_kwargs["crqs"]["net_type"] == "resnet":
            transform_net_create_fn = lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=base_kwargs["crqs"]["hidden_dim"],
                context_features=context_dim,
                num_blocks=base_kwargs["crqs"]["num_transform_blocks"],
                activation=activation,
                dropout_probability=base_kwargs["crqs"]["dropout_probability"],
                use_batch_norm=base_kwargs["crqs"]["batch_norm"],
            )
        elif base_kwargs["crqs"]["net_type"] == "mlp":
            transform_net_create_fn = lambda in_features, out_features: MLP(
                in_features,
                out_features,
                context_features=context_dim,
                hidden_sizes=base_kwargs["crqs"]["hidden_dim"],
                activation=base_kwargs["crqs"]["activation"],
                dropout_probability=base_kwargs["crqs"]["dropout_probability"],
                use_batch_norm=base_kwargs["crqs"]["batch_norm"],
            )
        else:
            raise ValueError("Unknown net_type")

        transform.append(
            PiecewiseRationalQuadraticCouplingTransformM(
                mask=utils.create_alternating_binary_mask(
                    features=input_dim, even=True
                ),
                transform_net_create_fn=transform_net_create_fn,
                num_bins=base_kwargs["crqs"]["num_bins"],
                tails="linear",
                tail_bound=base_kwargs["crqs"]["tail_bound"],
                apply_unconditional_transform=False,
                img_shape=(1, 28, 28),
                min_bin_width=base_kwargs["crqs"]["min_bin_width"],
                min_bin_height=base_kwargs["crqs"]["min_bin_height"],
                min_derivative=base_kwargs["crqs"]["min_derivative"],
            )
        )

        if base_kwargs["permute_type"] != "no-permutation":
            transform.append(create_random_transform(param_dim=input_dim))

    transform_fnal = transforms.CompositeTransform(transform)

    flow = FlowM(transform_fnal, distribution)

    # Store hyperparameters. This is for reconstructing model when loading from
    # saved file.

    flow.model_hyperparams = {
        "input_dim": input_dim,
        "context_dim": context_dim,
        "base_kwargs": base_kwargs,
    }

    return flow


def save_model(
    epoch,
    model,
    scheduler,
    train_history,
    test_history,
    name,
    model_dir=None,
    optimizer=None,
):
    """Save a model and optimizer to file.
    Args:
        model:      model to be saved
        optimizer:  optimizer to be saved
        epoch:      current epoch number
        model_dir:  directory to save the model in
        filename:   filename for saved model
    """

    if model_dir is None:
        raise NameError("Model directory must be specified.")

    filename = name + f"_@epoch_{epoch}.pt"
    resume_filename = "checkpoint-latest.pt"

    p = Path(model_dir)
    p.mkdir(parents=True, exist_ok=True)

    dict = {
        "train_history": train_history,
        "test_history": test_history,
        "model_hyperparams": model.model_hyperparams,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        dict["scheduler_state_dict"] = scheduler.state_dict()
        dict["last_lr"] = scheduler._last_lr

    torch.save(dict, p / filename)
    torch.save(dict, p / resume_filename)


def load_mixture_model(device, model_dir=None, filename=None):
    """Load a saved model.
    Args:
        filename:       File name
    """

    if model_dir is None:
        raise NameError(
            "Model directory must be specified."
            " Store in attribute PosteriorModel.model_dir"
        )

    p = Path(model_dir)
    checkpoint = torch.load(p / filename, map_location="cpu", weights_only=False)

    model_hyperparams = checkpoint["model_hyperparams"]
    # added because of a bug in the old create_mixture_flow_model function
    try:
        if checkpoint["model_hyperparams"]["base_transform_kwargs"] is not None:
            checkpoint["model_hyperparams"]["base_kwargs"] = checkpoint[
                "model_hyperparams"
            ]["base_transform_kwargs"]
            del checkpoint["model_hyperparams"]["base_transform_kwargs"]
    except KeyError:
        pass
    train_history = checkpoint["train_history"]
    test_history = checkpoint["test_history"]

    # Load model
    model = create_mixture_flow_model(**model_hyperparams)
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.to(device)

    # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model.eval()

    # Load optimizer
    scheduler_present_in_checkpoint = "scheduler_state_dict" in checkpoint.keys()

    # If the optimizer has more than 1 param_group, then we built it with
    # flow_lr different from lr
    if len(checkpoint["optimizer_state_dict"]["param_groups"]) > 1:
        flow_lr = checkpoint["last_lr"]
    elif checkpoint["last_lr"] is not None:
        flow_lr = checkpoint["last_lr"][0]
    else:
        flow_lr = None

    # Set the epoch to the correct value. This is needed to resume
    # training.
    epoch = checkpoint["epoch"]

    return (
        model,
        scheduler_present_in_checkpoint,
        flow_lr,
        epoch,
        train_history,
        test_history,
    )

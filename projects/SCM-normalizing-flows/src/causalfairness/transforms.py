# type: ignore
# type: ignore[C0301]
""" transform definition"""

from torch.distributions.utils import lazy_property
from torch.distributions import constraints
from torch.distributions.transforms import Transform
import torch
import numpy as np




class SqueezeTransform(Transform):
    """A transformation defined for image data that trades spatial dimensions for channel
    dimensions, i.e. "squeezes" the inputs along the channel dimensions.
    Implementation adapted from https://github.com/pclucas14/pytorch-glow and
    https://github.com/chaiyujin/glow-pytorch.
    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    codomain = constraints.real
    bijective = True
    event_dim = 3
    volume_preserving = True

    def __init__(self, factor=2):
        super().__init__(cache_size=1)

        self.factor = factor

    def _call(self, inputs):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor
        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        if inputs.dim() < 3:
            raise ValueError(
                f"Expecting inputs with at least 3 dimensions, got {inputs.shape} - {inputs.dim()}"
            )

        *batch_dims, c, h, w = inputs.size()
        num_batch = len(batch_dims)

        if h % self.factor != 0 or w % self.factor != 0:
            raise ValueError("Input image size not compatible with the factor.")

        inputs = inputs.view(
            *batch_dims, c, h // self.factor, self.factor, w // self.factor, self.factor
        )
        permute = np.array((0, 2, 4, 1, 3)) + num_batch
        inputs = inputs.permute(*np.arange(num_batch), *permute).contiguous()
        inputs = inputs.view(
            *batch_dims,
            c * self.factor * self.factor,
            h // self.factor,
            w // self.factor,
        )

        return inputs

    def _inverse(self, inputs):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x.
        """
        if inputs.dim() < 3:
            raise ValueError(
                f"Expecting inputs with at least 3 dimensions, got {inputs.shape}"
            )

        *batch_dims, c, h, w = inputs.size()
        num_batch = len(batch_dims)

        if c < 4 or c % 4 != 0:
            raise ValueError("Invalid number of channel dimensions.")

        inputs = inputs.view(
            *batch_dims, c // self.factor ** 2, self.factor, self.factor, h, w
        )
        permute = np.array((0, 3, 1, 4, 2)) + num_batch
        inputs = inputs.permute(*np.arange(num_batch), *permute).contiguous()
        inputs = inputs.view(
            *batch_dims, c // self.factor ** 2, h * self.factor, w * self.factor
        )

        return inputs

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])). Note that this type of
        transform is not autoregressive, so the log Jacobian is not the sum of the
        previous expression. However, it turns out it's always 0 (since the
        determinant is -1 or +1), and so returning a vector of zeros works.
        """

        log_abs_det_jacobian = torch.zeros(
            x.size()[:-3], dtype=x.dtype, layout=x.layout, device=x.device
        )
        return log_abs_det_jacobian

    def get_output_shape(self, c, h, w):
        """ returns output shape"""
        return (c * self.factor * self.factor, h // self.factor, w // self.factor)


class ReshapeTransform(Transform):
    codomain = constraints.real
    bijective = True
    volume_preserving = True

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.event_dim = len(input_shape)
        self.inv_event_dim = len(output_shape)
        self.input_shape = input_shape
        self.output_shape = output_shape

    def _call(self, inputs):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor
        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        batch_dims = inputs.shape[: -self.event_dim]
        inp_shape = inputs.shape[-self.event_dim :]
        if inp_shape != self.input_shape:
            raise RuntimeError(
                "Unexpected inputs shape ({}, but expecting {})".format(
                    inp_shape, self.input_shape
                )
            )
        return inputs.reshape(*batch_dims, *self.output_shape)

    def _inverse(self, inputs):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x.
        """
        batch_dims = inputs.shape[: -self.inv_event_dim]
        inp_shape = inputs.shape[-self.inv_event_dim :]
        if inp_shape != self.output_shape:
            raise RuntimeError(
                "Unexpected inputs shape ({}, but expecting {})".format(
                    inp_shape, self.output_shape
                )
            )
        return inputs.reshape(*batch_dims, *self.input_shape)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])). Note that this type of
        transform is not autoregressive, so the log Jacobian is not the sum of the
        previous expression. However, it turns out it's always 0 (since the
        determinant is -1 or +1), and so returning a vector of zeros works.
        """

        log_abs_det_jacobian = torch.zeros(
            x.size()[: -self.event_dim], dtype=x.dtype, layout=x.layout, device=x.device
        )
        return log_abs_det_jacobian


class TransposeTransform(Transform):
    """
    A bijection that reorders the input dimensions, that is, multiplies the input by
    a permutation matrix. This is useful in between
    :class:`~pyro.distributions.transforms.AffineAutoregressive` transforms to
    increase the flexibility of the resulting distribution and stabilize learning.
    Whilst not being an autoregressive transform, the log absolute determinate of
    the Jacobian is easily calculable as 0. Note that reordering the input dimension
    between two layers of
    :class:`~pyro.distributions.transforms.AffineAutoregressive` is not equivalent
    to reordering the dimension inside the MADE networks that those IAFs use; using
    a :class:`~pyro.distributions.transforms.Permute` transform results in a
    distribution with more flexibility.
    Example usage:
    >>> from pyro.nn import AutoRegressiveNN
    >>> from pyro.distributions.transforms import AffineAutoregressive, Permute
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> iaf1 = AffineAutoregressive(AutoRegressiveNN(10, [40]))
    >>> ff = Permute(torch.randperm(10, dtype=torch.long))
    >>> iaf2 = AffineAutoregressive(AutoRegressiveNN(10, [40]))
    >>> flow_dist = dist.TransformedDistribution(base_dist, [iaf1, ff, iaf2])
    >>> flow_dist.sample()  # doctest: +SKIP
    :param permutation: a permutation ordering that is applied to the inputs.
    :type permutation: torch.LongTensor
    """

    codomain = constraints.real
    bijective = True
    volume_preserving = True

    def __init__(self, permutation):
        super().__init__(cache_size=1)

        self.event_dim = len(permutation)
        self.permutation = permutation

    @lazy_property
    def inv_permutation(self):
        result = torch.empty_like(self.permutation, dtype=torch.long)
        result[self.permutation] = torch.arange(
            self.permutation.size(0), dtype=torch.long, device=self.permutation.device
        )
        return result

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor
        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """

        *batch_dims, c, h, w = x.size()
        num_batch = len(batch_dims)

        return x.permute(
            *np.arange(num_batch), *(self.permutation + num_batch)
        ).contiguous()

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x.
        """

        *batch_dims, c, h, w = y.size()
        num_batch = len(batch_dims)

        return y.permute(
            *np.arange(num_batch), *(self.inv_permutation + num_batch)
        ).contiguous()

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])). Note that this type of
        transform is not autoregressive, so the log Jacobian is not the sum of the
        previous expression. However, it turns out it's always 0 (since the
        determinant is -1 or +1), and so returning a vector of zeros works.
        """

        log_abs_det_jacobian = torch.zeros(
            x.size()[: -self.event_dim], dtype=x.dtype, layout=x.layout, device=x.device
        )
        return log_abs_det_jacobian


from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions import transforms as pyro_transforms
from torch.distributions import transforms

import torch


class LearnedAffineTransform(TransformModule, transforms.AffineTransform):
    def __init__(self, loc=None, scale=None, **kwargs):

        super().__init__(loc=loc, scale=scale, **kwargs)

        if loc is None:
            self.loc = torch.nn.Parameter(
                torch.zeros(
                    [
                        1,
                    ]
                )
            )
        if scale is None:
            self.scale = torch.nn.Parameter(
                torch.ones(
                    [
                        1,
                    ]
                )
            )

    def _broadcast(self, val):
        dim_extension = tuple(1 for _ in range(val.dim() - 1))
        loc = self.loc.view(-1, *dim_extension)
        scale = self.scale.view(-1, *dim_extension)

        return loc, scale

    def _call(self, x):
        loc, scale = self._broadcast(x)

        return loc + scale * x

    def _inverse(self, y):
        loc, scale = self._broadcast(y)
        return (y - loc) / scale


class ConditionalAffineTransform(ConditionalTransformModule):
    def __init__(self, context_nn, event_dim=0, **kwargs):
        super().__init__(**kwargs)

        self.event_dim = event_dim
        self.context_nn = context_nn

    def condition(self, context):
        loc, log_scale = self.context_nn(context)
        scale = torch.exp(log_scale)

        ac = transforms.AffineTransform(loc, scale, event_dim=self.event_dim)
        return ac


class LowerCholeskyAffine(pyro_transforms.LowerCholeskyAffine):
    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs(dy/dx)).
        """
        return torch.ones(
            x.size()[:-1], dtype=x.dtype, layout=x.layout, device=x.device
        ) * self.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1).sum(-1)


class ActNorm(TransformModule):
    codomain = constraints.real
    bijective = True
    event_dim = 3

    def __init__(self, features):
        """
        Transform that performs activation normalization. Works for 2D and 4D inputs. For 4D
        inputs (images) normalization is performed per-channel, assuming BxCxHxW input shape.
        Reference:
        > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
        """

        self.initialized = False
        self.log_scale = nn.Parameter(torch.zeros(features))
        self.shift = nn.Parameter(torch.zeros(features))

    @property
    def scale(self):
        """ returns scale"""
        return torch.exp(self.log_scale)

    def _broadcastable_scale_shift(self, inputs):
        if inputs.dim() == 4:
            return self.scale.view(1, -1, 1, 1), self.shift.view(1, -1, 1, 1)
        else:
            return self.scale.view(1, -1), self.shift.view(1, -1)

    def _call(self, x):
        if x.dim() not in [2, 4]:
            raise ValueError("Expecting inputs to be a 2D or a 4D tensor.")

        if self.training and not self.initialized:
            self._initialize(x)

        scale, shift = self._broadcastable_scale_shift(x)
        outputs = scale * x + shift

        return outputs

    def _inverse(self, y):
        if y.dim() not in [2, 4]:
            raise ValueError("Expecting inputs to be a 2D or a 4D tensor.")

        scale, shift = self._broadcastable_scale_shift(y)
        outputs = (y - shift) / scale

        return outputs

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])). Note that this type of
        transform is not autoregressive, so the log Jacobian is not the sum of the
        previous expression. However, it turns out it's always 0 (since the
        determinant is -1 or +1), and so returning a vector of zeros works.
        """

        ones = torch.ones(x.shape[0], device=x.device)
        if x.dim() == 4:
            _, _, h, w = x.shape
            log_abs_det_jacobian = h * w * torch.sum(self.log_scale) * ones
        else:
            log_abs_det_jacobian = torch.sum(self.log_scale) * ones

        return log_abs_det_jacobian

    def _initialize(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance."""
        if inputs.dim() == 4:
            num_channels = inputs.shape[1]
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, num_channels)

        with torch.no_grad():
            std = inputs.std(dim=0)
            mu = (inputs / std).mean(dim=0)
            self.log_scale.data = -torch.log(std)
            self.shift.data = -mu

        self.initialized = True

""" Contains model definitions """

import torch
import pyro
from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.conditional import ConditionalTransformedDistribution
from pyro.distributions.transforms import (
    Spline, ComposeTransform, ConditionalAffineCoupling,
    GeneralizedChannelPermute, SigmoidTransform
    )
from pyro.nn import DenseNN
from src.normalizingFlowsSCM.transforms import (
    ReshapeTransform, SqueezeTransform,
    TransposeTransform,
    ConditionalAffineTransform, ActNorm
    )
from src.normalizingFlowsSCM.arch import BasicFlowConvNet


class FlowSCM(pyroModule):
    """ definition of FlowSCM class"""
    def __init__(self, use_affine_ex=True, **kwargs):
        super.__init__(**kwargs)

        self.num_scales = 2

        self.register_buffer("glasses_base_loc", torch.zeros([1, ], requires_grad=False))
        self.register_buffer("glasses_base_scale", torch.ones([1, ], requires_grad=False))

        self.register_buffer("glasses_flow_lognorm_loc", torch.zeros([], requires_grad=False))
        self.register_buffer("glasses_flow_lognorm_scale", torch.ones([], requires_grad=False))

        self.glasses_flow_components = ComposeTransformModule([Spline(1)])
        self.glasses_flow_constraint_transforms = ComposeTransform([self.glasses_flow_lognorm,
            SigmoidTransform()])
        self.glasses_flow_transforms = ComposeTransform([self.glasses_flow_components,
            self.glasses_flow_constraint_transforms])

        glasses_base_dist = Normal(self.glasses_base_loc, self.glasses_base_scale).to_event(1)
        self.glasses_dist = TransformedDistribution(glasses_base_dist, self.glasses_flow_transforms)
        glasses_ = pyro.sample("glasses_", self.glasses_dist)
        glasses = pyro.sample("glasses", dist.Bernoulli(glasses_))
        glasses_context = self.glasses_flow_constraint_transforms.inv(glasses_)

        self.x_transforms = self._build_image_flow()
        self.register_buffer("x_base_loc", torch.zeros([1, 64, 64], requires_grad=False))
        self.register_buffer("x_base_scale", torch.ones([1, 64, 64], requires_grad=False))
        x_base_dist = Normal(self.x_base_loc, self.x_base_scale).to_event(3)
        cond_x_transforms = ComposeTransform(
            ConditionalTransformedDistribution(x_base_dist, self.x_transforms)
            .condition(context).transforms
            ).inv
        cond_x_dist = TransformedDistribution(x_base_dist, cond_x_transforms)

        x = pyro.sample("x", cond_x_dist)

        return x, glasses


    def _build_image_flow(self):
        self.trans_modules = ComposeTransformModule([])
        self.x_transforms = []
        self.x_transforms += [self._get_preprocess_transforms()]

        c = 1
        for _ in range(self.num_scales):
            self.x_transforms.append(SqueezeTransform())
            c *= 4

            for _ in range(self.flows_per_scale):
                if self.use_actnorm:
                    actnorm = ActNorm(c)
                    self.trans_modules.append(actnorm)
                    self.x_transforms.append(actnorm)

                gcp = GeneralizedChannelPermute(channels=c)
                self.trans_modules.append(gcp)
                self.x_transforms.append(gcp)

                self.x_transforms.append(TransposeTransform(torch.tensor((1, 2, 0))))

                ac = ConditionalAffineCoupling(c // 2,
                                               BasicFlowConvNet(c // 2,
                                                self.hidden_channels, (c // 2, c // 2), 2))
                self.trans_modules.append(ac)
                self.x_transforms.append(ac)

                self.x_transforms.append(TransposeTransform(torch.tensor((2, 0, 1))))

            gcp = GeneralizedChannelPermute(channels=c)
            self.trans_modules.append(gcp)
            self.x_transforms.append(gcp)

        self.x_transforms += [
            ReshapeTransform((4**self.num_scales, 32 // 2**self.num_scales, 32 // 2**self.num_scales), (1, 32, 32))
        ]

        if self.use_affine_ex:
            affine_net = DenseNN(2, [16, 16], param_dims=[1, 1])
            affine_trans = ConditionalAffineTransform(context_nn=affine_net, event_dim=3)

            self.trans_modules.append(affine_trans)
            self.x_transforms.append(affine_trans)



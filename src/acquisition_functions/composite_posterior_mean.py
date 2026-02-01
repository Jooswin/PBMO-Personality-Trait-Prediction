from __future__ import annotations

from typing import List

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import (
    t_batch_mode_transform,
)
from torch import Tensor


class CompositePosteriorMean(AcquisitionFunction):
    r"""Analytic Expected Utility of Best Option (EUBO)"""

    def __init__(
        self,
        attribute_models: List[Model],
        utility_model: Model,
        attribute_lower_bounds: Tensor,
        attribute_upper_bounds: Tensor,
    ) -> None:
        r""".

        .

        Args:
            attribute_models: .
            utility_model: .
        """
        super().__init__(model=utility_model)
        self.attribute_models = attribute_models
        self.utility_model = utility_model
        self.attribute_lower_bounds = attribute_lower_bounds
        self.attribute_upper_bounds = attribute_upper_bounds

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r""".
        Args:
            X: A `batch_shape x 1 x d`-dim Tensor.
        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape`.
        """
        attributes_mean = []

        for attribute_model in self.attribute_models:
            attribute_posterior = attribute_model.posterior(X)
            attributes_mean.append(attribute_posterior.mean)
        attributes_mean = torch.cat(attributes_mean, dim=-1)
        normalized_attributes_mean = (attributes_mean - self.attribute_lower_bounds) / (
            self.attribute_upper_bounds - self.attribute_lower_bounds
        )
        acqf_val = (
            self.utility_model.posterior(normalized_attributes_mean)
            .mean.squeeze(-1)
            .squeeze(-1)
        )
        return acqf_val

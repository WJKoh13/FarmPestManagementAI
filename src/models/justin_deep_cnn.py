"""Justin's scratch-built deeper sequential CNN for IP102 classification.

This model is the team's second controlled experiment. Compared with the
three-block baseline, it adds a second convolution at every resolution and a
fourth feature stage. It uses only primitive PyTorch layers, starts from random
weights, and does not import a pretrained or prebuilt CNN architecture.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


class ConvStage(nn.Sequential):
    """Two convolutional layers followed by downsampling and optional dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))

        super().__init__(*layers)


class JustinDeepCNN(nn.Module):
    """Four-stage sequential CNN for fine-grained pest classification.

    Args:
        num_classes: Number of output classes.
        classifier_dropout: Dropout probability before the final classifier.
        stage_dropouts: Dropout probabilities after each of the four feature
            stages. Keeping this configurable lets experiments change
            regularization without changing the CNN architecture.

    Input:
        A float tensor shaped ``[batch_size, 3, height, width]``. The shared
        experiment uses 160 by 160 RGB images.

    Output:
        Raw logits shaped ``[batch_size, num_classes]``. Softmax is not applied
        because ``torch.nn.CrossEntropyLoss`` expects raw logits.
    """

    def __init__(
        self,
        num_classes: int = 10,
        classifier_dropout: float = 0.40,
        stage_dropouts: Sequence[float] = (0.00, 0.10, 0.20, 0.30),
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0.0 <= classifier_dropout < 1.0:
            raise ValueError(
                "classifier_dropout must be in the range [0.0, 1.0)"
            )
        if len(stage_dropouts) != 4:
            raise ValueError("stage_dropouts must contain exactly four values")
        if any(not 0.0 <= dropout < 1.0 for dropout in stage_dropouts):
            raise ValueError(
                "every stage dropout must be in the range [0.0, 1.0)"
            )

        self.num_classes = num_classes
        self.stage_dropouts = tuple(float(value) for value in stage_dropouts)

        self.features = nn.Sequential(
            ConvStage(3, 32, dropout=self.stage_dropouts[0]),
            ConvStage(32, 64, dropout=self.stage_dropouts[1]),
            ConvStage(64, 128, dropout=self.stage_dropouts[2]),
            ConvStage(128, 256, dropout=self.stage_dropouts[3]),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(in_features=256, out_features=num_classes),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize all trainable layers without loading external weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, images: Tensor) -> Tensor:
        """Extract one 256-value feature vector per input image."""
        features = self.features(images)
        pooled = self.global_pool(features)
        return pooled.flatten(start_dim=1)

    def forward(self, images: Tensor) -> Tensor:
        """Return raw class logits for a batch of RGB images."""
        features = self.features(images)
        pooled = self.global_pool(features)
        return self.classifier(pooled)


def build_model(
    num_classes: int = 10,
    classifier_dropout: float = 0.40,
    stage_dropouts: Sequence[float] = (0.00, 0.10, 0.20, 0.30),
) -> JustinDeepCNN:
    """Backward-compatible builder used by the original notebook."""
    return JustinDeepCNN(
        num_classes=num_classes,
        classifier_dropout=classifier_dropout,
        stage_dropouts=stage_dropouts,
    )


def build_justin_deep_cnn(
    num_classes: int = 10,
    classifier_dropout: float = 0.20,
    stage_dropouts: Sequence[float] = (0.00, 0.00, 0.05, 0.10),
) -> JustinDeepCNN:
    """Build the registered randomly initialized deep CNN."""
    return build_model(
        num_classes=num_classes,
        classifier_dropout=classifier_dropout,
        stage_dropouts=stage_dropouts,
    )


if __name__ == "__main__":
    import torch

    model = build_model()
    example_batch = torch.randn(2, 3, 160, 160)
    output = model(example_batch)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(model)
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Trainable parameters: {parameter_count:,}")

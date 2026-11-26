from copy import deepcopy
import re

import torch
from torch import nn

from m5.model.flavors import FLAVORS


def depth_logits(model, hidden, codes):
    """At position i (1–15), predict CB i from temporal context and CB < i."""
    prefix = [model._embed_audio(i, codes[:, i]) for i in range(codes.shape[1] - 1)]
    projected = model.projection(torch.stack([hidden, *prefix], dim=1))
    pos = torch.arange(projected.shape[1], device=codes.device)[None]
    depth = model.decoder(projected, None, pos)
    return torch.einsum("bkd,kdv->bkv", depth[:, 1:], model.audio_head)


def register_flavor(name):
    """Register a self-describing pruned architecture for upstream checkpoint loading."""
    match = re.fullmatch(r"lite-(\d+)-(\d+)-(.+)", name)
    if match is None:
        return
    layers, width, base = match.groups()
    layers, width = int(layers), int(width)
    base_factory = FLAVORS[base]

    def factory(attention):
        decoder = base_factory(attention)
        if not 0 < layers <= len(decoder.layers) or not 0 < width <= decoder.config.intermediate_size:
            raise ValueError(f"Invalid Lite decoder dimensions: {name}")
        decoder.layers = decoder.layers[:layers]
        for layer in decoder.layers:
            mlp = layer.mlp
            hidden = decoder.config.hidden_size
            mlp.gate_proj = nn.Linear(hidden, width, bias=mlp.gate_proj.bias is not None)
            mlp.up_proj = nn.Linear(hidden, width, bias=mlp.up_proj.bias is not None)
            mlp.down_proj = nn.Linear(width, hidden, bias=mlp.down_proj.bias is not None)
            mlp.intermediate_size = width
        decoder.num_layers = decoder.config.num_hidden_layers = layers
        decoder.config.intermediate_size = width
        return decoder

    FLAVORS[name] = factory


@torch.no_grad()
def prune(model, layers, width):
    teacher = model.decoder
    if not layers or layers != sorted(set(layers)) or not 0 <= layers[0] <= layers[-1] < len(teacher.layers):
        raise ValueError("Layer indices must be ascending, unique and in range")
    base = model.args.decoder_flavor
    name = f"lite-{len(layers)}-{width}-{base}"
    register_flavor(name)
    student = FLAVORS[name](teacher.config._attn_implementation)
    reference = next(teacher.parameters())
    student.to(device=reference.device, dtype=reference.dtype)
    student.norm.load_state_dict(teacher.norm.state_dict())
    for target, index in zip(student.layers, layers):
        source = teacher.layers[index]
        weights = deepcopy(source.state_dict())
        gate, up, down = (weights[f"mlp.{part}_proj.weight"] for part in ("gate", "up", "down"))
        score = gate.float().norm(dim=1) * up.float().norm(dim=1) * down.float().norm(dim=0)
        channels = score.topk(width).indices.sort().values
        for part in ("gate_proj", "up_proj"):
            for suffix in ("weight", "bias"):
                key = f"mlp.{part}.{suffix}"
                if key in weights:
                    weights[key] = weights[key][channels]
        weights["mlp.down_proj.weight"] = down[:, channels]
        target.load_state_dict(weights)
    model.requires_grad_(False)
    model.decoder = student
    model.args.decoder_flavor = name
    return model

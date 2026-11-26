import argparse
import json
import math
from pathlib import Path
import shutil

import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from .common import activate_upstream, bundle_id, digest, new_directory, read_json, records, save_json


DEFAULTS = {
    "layers": [0, 2, 4], "ffn_width": 1024, "epochs": 5, "batch_frames": 64,
    "learning_rate": 0.0001, "kd_weight": 0.7, "seed": 42,
}


def losses(logits, batch, temperature):
    logits = logits.float()
    indices = batch["indices"].long()
    scaled = logits / temperature
    log_norm = scaled.logsumexp(dim=-1)
    log_q = scaled.gather(-1, indices) - log_norm.unsqueeze(-1)
    log_tail = scaled.scatter(-1, indices, -torch.inf).logsumexp(dim=-1) - log_norm
    p, tail = batch["prob"].float(), batch["tail"].float()
    kl = (torch.special.xlogy(p, p) - p * log_q).sum(-1)
    kl = (kl + torch.special.xlogy(tail, tail) - tail * log_tail).mean()
    ce = F.cross_entropy(logits.flatten(0, 1), batch["codes"][:, 1:].long().flatten())
    return kl, ce


def batches(path, metadata, batch_frames, seed=None):
    random = torch.Generator()
    shards = metadata["shards"]
    order = range(len(shards)) if seed is None else torch.randperm(len(shards), generator=random.manual_seed(seed))
    for index in order:
        shard = shards[index]
        tensors = load_file(str(path / shard["file"]))
        n = shard["frames"]
        if n <= 0 or any(value.shape[0] != n for value in tensors.values()):
            raise ValueError(f"Invalid shard length: {shard['file']}")
        rows = torch.arange(n) if seed is None else torch.randperm(n, generator=random)
        for chosen in rows.split(batch_frames):
            yield {key: value[chosen] for key, value in tensors.items()}


def pass_once(model, path, metadata, config, device, optimizer=None, epoch=0):
    from .depth import depth_logits

    model.eval()
    model.decoder.train(optimizer is not None)
    total = torch.zeros(3, device=device)
    frames = 0
    seed = config["seed"] + epoch if optimizer is not None else None
    with torch.set_grad_enabled(optimizer is not None):
        for batch in batches(path, metadata, config["batch_frames"], seed):
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = depth_logits(model, batch["hidden"], batch["codes"].long())
            temperature = metadata["temperature"]
            kl, ce = losses(logits, batch, temperature)
            weight = config["kd_weight"]
            loss = weight * temperature ** 2 * kl + (1 - weight) * ce
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            count = len(logits)
            total += torch.stack([loss.detach(), kl.detach(), ce.detach()]) * count
            frames += count
    if frames != metadata["frames"] or not frames or not torch.isfinite(total).all():
        raise ValueError(f"Invalid frame count or loss: {path}")
    return dict(zip(("loss", "kl", "ce"), (total / frames).tolist()))


def write_bundle(path, teacher, model, optimizer, state):
    with new_directory(path) as staging:
        shutil.copyfile(teacher / "conditioning.json", staging / "conditioning.json")
        shutil.copytree(teacher / "tokenizer", staging / "tokenizer")
        config = json.loads(read_json(teacher / "config.json"))
        config["decoder_flavor"] = model.args.decoder_flavor
        save_json(staging / "config.json", config)
        save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
                  str(staging / "model.safetensors"))
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        save_json(staging / "distill.json", state)


def train(args):
    activate_upstream(args.upstream)
    from checkpoint import load_checkpoint
    from .depth import prune, register_flavor

    config = DEFAULTS | json.loads(read_json(args.config))
    if config.keys() != DEFAULTS.keys():
        raise ValueError(f"Unknown config fields: {config.keys() - DEFAULTS.keys()}")
    if min(config["epochs"], config["batch_frames"], config["ffn_width"]) < 1:
        raise ValueError("epochs, batch_frames and ffn_width must be positive")
    if not 0 <= config["kd_weight"] <= 1 or not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("Invalid KD weight or learning rate")
    torch.manual_seed(config["seed"])
    train_meta, val_meta = [json.loads(read_json(path / "tracks.json")) for path in (args.train, args.val)]
    teacher_id = bundle_id(args.teacher)
    for meta, split in ((train_meta, "train"), (val_meta, "val")):
        if meta["version"] != 1 or meta["teacher"] != teacher_id or meta["split"] != [split]:
            raise ValueError(f"Incompatible teacher or split in {split} tracks")
    for field in ("codec", "temperature", "topk"):
        if train_meta[field] != val_meta[field]:
            raise ValueError(f"Track metadata mismatch: {field}")
    train_ids = {row["utt"] for row in records(args.train / "utterances.jsonl")}
    if any(row["utt"] in train_ids for row in records(args.val / "utterances.jsonl")):
        raise ValueError("Train and validation overlap")
    provenance = {
        "train": digest(args.train / "tracks.json"), "val": digest(args.val / "tracks.json"),
        "config": {key: value for key, value in config.items() if key != "epochs"},
    }
    start, best = 0, float("inf")
    if args.resume:
        state = json.loads(read_json(args.resume / "distill.json"))
        if state["provenance"] != provenance:
            raise ValueError("Resume requires the same tracks and training settings")
        register_flavor(json.loads(read_json(args.resume / "config.json"))["decoder_flavor"])
        model, _ = load_checkpoint(args.resume)
        model.requires_grad_(False)
        model.decoder.requires_grad_(True)
        start, best = state["epoch"], state["best"]
    else:
        model, _ = load_checkpoint(args.teacher)
        teacher_depth = sum(p.numel() for p in model.decoder.parameters())
        prune(model, config["layers"], config["ffn_width"])
        print(json.dumps({"teacher_depth_parameters": teacher_depth,
                          "student_depth_parameters": sum(p.numel() for p in model.decoder.parameters())}),
              flush=True)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config["learning_rate"],
                                  betas=(0.9, 0.95), weight_decay=0.01)
    if args.resume:
        optimizer.load_state_dict(torch.load(args.resume / "optimizer.pt", map_location="cpu", weights_only=True))
    if start >= config["epochs"]:
        raise ValueError("epochs must exceed the resume checkpoint epoch")
    if not args.resume:
        baseline = pass_once(model, args.val, val_meta, config, args.device)
        print(json.dumps({"epoch": 0, "val": baseline}), flush=True)
        write_bundle(args.output / "epoch-0000", args.teacher, model, optimizer,
                     {"epoch": 0, "val": baseline, "best": baseline["loss"],
                      "is_best": True, "provenance": provenance, "codec": train_meta["codec"]})
        best = baseline["loss"]
    for epoch in range(start + 1, config["epochs"] + 1):
        training = pass_once(model, args.train, train_meta, config, args.device, optimizer, epoch)
        validation = pass_once(model, args.val, val_meta, config, args.device)
        is_best = validation["loss"] < best
        best = min(best, validation["loss"])
        state = dict(epoch=epoch, train=training, val=validation, best=best, is_best=is_best,
                     provenance=provenance, codec=train_meta["codec"])
        write_bundle(args.output / f"epoch-{epoch:04d}", args.teacher, model, optimizer, state)
        print(json.dumps({key: state[key] for key in ("epoch", "train", "val", "is_best")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train the pruned M5 Depth Decoder")
    for name in ("upstream", "teacher", "train", "val", "config", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    train(parser.parse_args())


if __name__ == "__main__":
    main()

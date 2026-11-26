import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

from .common import activate_upstream, bundle_id, digest, new_directory, records, save_json


def summarize(logits, k, temperature):
    logits = logits.float() / temperature
    values, indices = logits.topk(k, dim=-1)
    normalizer = logits.logsumexp(dim=-1)
    remainder = logits.scatter(-1, indices, -torch.inf).logsumexp(dim=-1)
    return {
        "indices": indices.to(torch.int16),
        "prob": (values - normalizer.unsqueeze(-1)).exp(),
        "tail": (remainder - normalizer).exp(),
    }


@torch.inference_mode()
def extract(args):
    activate_upstream(args.upstream)
    import soundfile as sf
    import torchaudio
    from checkpoint import load_checkpoint
    from generator import Generator
    from m5.training.text import Segment, _tokenize_segment
    from .depth import depth_logits

    if not 1 <= args.topk < 2051 or not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("Expected 1 <= topk < 2051 and finite temperature > 0")
    if not 0 < args.batch_frames <= args.shard_frames:
        raise ValueError("Expected 0 < batch_frames <= shard_frames")

    teacher_id = bundle_id(args.teacher)
    codec_id = {item.name: digest(item) for item in sorted(args.codec.iterdir())
                if item.suffix in (".json", ".safetensors") and item.is_file()}
    model, tokenizer = load_checkpoint(args.teacher)
    model.to(args.device)
    generator = Generator(model, 16, tokenizer, args.codec, local_files_only=True)
    device = generator.device
    pending = []
    count = total = 0
    shards = []
    utterances = set()
    splits = set()

    def flush():
        nonlocal count
        if not count:
            return
        name = f"{len(shards):06d}.safetensors"
        save_file({key: torch.cat([part[key] for part in pending]) for key in pending[0]},
                  str(staging / name))
        shards.append({"file": name, "frames": count})
        pending.clear()
        count = 0

    with new_directory(args.output) as staging:
        with (staging / "utterances.jsonl").open("w", encoding="utf-8") as index:
            for row in records(args.manifest):
                if row["utt"] in utterances:
                    raise ValueError(f"Duplicate utterance: {row['utt']}")
                utterances.add(row["utt"])
                splits.add(row["split"])
                path = Path(row["audio"])
                if digest(path) != row["sha256"]:
                    raise ValueError(f"Audio changed after manifest preparation: {path}")
                audio, rate = sf.read(path, dtype="float32", always_2d=True)
                audio = torch.from_numpy(audio.mean(axis=1))
                if not audio.numel() or not torch.isfinite(audio).all() or audio.abs().max() == 0:
                    raise ValueError(f"Invalid or silent recording: {path}")
                if rate != generator.sample_rate:
                    audio = torchaudio.functional.resample(audio, rate, generator.sample_rate)
                codes = generator._audio_tokenizer.encode(audio.numpy(), sr=generator.sample_rate).audio_codes[0]
                codes = codes.to(dtype=torch.long, device="cpu")
                if codes.ndim != 2 or codes.shape[1] != 16 or not codes.shape[0] or codes.min() < 0 or codes.max() >= 2048:
                    raise ValueError(f"Unexpected codec outputs: {row['utt']}")
                segment = Segment(
                    text=row["text"], audio=codes.T + 1,
                    lang="English" if row["lang"] == "en" else "Chinese",
                    speaker=row.get("conditioning_speaker") or 0,
                    accent=row.get("accent") or None, caption=row.get("caption") or None,
                )
                tokens, mask = _tokenize_segment(segment, generator._text_tokenizer, audio_num_codebooks=16)
                tokens, mask = tokens[None].to(device), mask[None].to(device)
                if tokens.shape[1] > model.backbone.config.max_position_embeddings:
                    raise ValueError(f"Utterance exceeds context limit: {row['utt']}")
                past, past_mask = tokens[:, :-1], mask[:, :-1]
                embeddings = (model._embed_tokens(past) * past_mask.unsqueeze(-1)).sum(dim=2)
                pos = torch.arange(past.shape[1], device=device)[None]
                hidden = model.backbone(embeddings, None, pos)
                # CB0=0 is EOS. Upstream generation uses it to decide when to stop.
                valid = mask[:, 1:, 0] & (tokens[:, 1:, 0] != 0)
                hidden, targets = hidden[valid], tokens[:, 1:, :16][valid]
                if len(targets) != len(codes):
                    raise ValueError(f"Frame alignment mismatch: {row['utt']}")
                index.write(json.dumps({"utt": row["utt"], "lang": row["lang"], "split": row["split"],
                                        "start": total, "frames": len(targets)}, ensure_ascii=False) + "\n")
                total += len(targets)
                for h, c in zip(hidden.split(args.batch_frames), targets.split(args.batch_frames)):
                    if count + len(c) > args.shard_frames:
                        flush()
                    soft = summarize(depth_logits(model, h, c), args.topk, args.temperature)
                    pending.append({key: value.cpu() for key, value in soft.items()} |
                                   {"hidden": h.float().cpu(), "codes": c.to(torch.int16).cpu()})
                    count += len(c)
        flush()
        if not shards:
            raise ValueError("Manifest yielded no audio frames")
        save_json(staging / "tracks.json", {
            "version": 1, "teacher": teacher_id, "codec": codec_id, "manifest": digest(args.manifest),
            "split": sorted(splits), "temperature": args.temperature, "topk": args.topk, "frames": total, "shards": shards,
        })
    print(f"Wrote {total} frames in {len(shards)} shards")


def main():
    parser = argparse.ArgumentParser(description="Cache teacher-forced M5 Depth targets")
    for name in ("upstream", "teacher", "codec", "manifest", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=2)
    parser.add_argument("--batch-frames", type=int, default=64)
    parser.add_argument("--shard-frames", type=int, default=5120)
    extract(parser.parse_args())


if __name__ == "__main__":
    main()

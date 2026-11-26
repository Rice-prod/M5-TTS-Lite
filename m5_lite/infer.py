import argparse
import json
from pathlib import Path
import time

import torch

from .common import activate_upstream, digest, read_json, save_json


def register_bundle(path):
    from .depth import register_flavor

    register_flavor(json.loads(read_json(path / "config.json"))["decoder_flavor"])


def main():
    parser = argparse.ArgumentParser(description="Synthesize WAV with a local M5 Lite bundle")
    for name in ("upstream", "model", "codec", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--text", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt-wav", type=Path)
    mode.add_argument("--speaker-id", type=int)
    parser.add_argument("--prompt-text")
    parser.add_argument("--lang", choices=("Chinese", "English"), default="Chinese")
    parser.add_argument("--accent")
    parser.add_argument("--caption")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-audio-length-ms", type=int, default=40000)
    args = parser.parse_args()
    if bool(args.prompt_wav) != bool(args.prompt_text):
        parser.error("--prompt-wav and --prompt-text must be provided together")
    if args.output.exists() or args.output.with_suffix(".stats.json").exists():
        parser.error("output WAV or statistics file already exists")
    if not args.codec.is_dir():
        parser.error("--codec must point to a local codec bundle")

    activate_upstream(args.upstream)
    from inference import M5TTS

    register_bundle(args.model)
    state = args.model / "distill.json"
    if state.is_file():
        expected_codec = json.loads(read_json(state))["codec"]
        for name, sha256 in expected_codec.items():
            if digest(args.codec / name) != sha256:
                raise ValueError(f"Codec does not match training tracks: {name}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    tts = M5TTS.from_pretrained(args.model, codec=args.codec, device=args.device,
                               attention="sdpa", local_files_only=True)
    start = time.perf_counter()
    audio = tts.infer(
        args.text, prompt_wav=args.prompt_wav, prompt_text=args.prompt_text, speaker_id=args.speaker_id,
        lang=args.lang, accent=args.accent, caption=args.caption, topk=args.topk,
        temperature=args.temperature, max_audio_length_ms=args.max_audio_length_ms,
    )
    elapsed = time.perf_counter() - start
    seconds = audio.numel() / tts.sample_rate
    tts.save(args.output, audio)
    stats = {"seconds": seconds, "inference_seconds": elapsed, "rtf": elapsed / seconds,
             "sample_rate": tts.sample_rate, "device": args.device, "threads": args.threads,
             "seed": args.seed, "topk": args.topk, "temperature": args.temperature}
    save_json(args.output.with_suffix(".stats.json"), stats)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .common import digest, new_directory, save_json


QUOTAS = {"zh": 0.6, "en": 0.3, "mix": 0.1}
SPLITS = ("train", "val", "test")


def prepare(csv_path, output, seed):
    import soundfile as sf

    groups, speakers, audio_hashes = {}, {}, {}
    utterances = set()
    splits = {split: [] for split in SPLITS}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            for key in ("utt", "audio", "text", "lang", "speaker", "group", "split", "license"):
                if not row.get(key, "").strip():
                    raise ValueError(f"Missing {key} in CSV record: {row}")
                row[key] = row[key].strip()
            utt, split = row["utt"], row["split"]
            if utt in utterances or split not in splits or row["lang"] not in QUOTAS:
                raise ValueError(f"Invalid ID, split or language: {utt}")
            utterances.add(utt)
            path = (csv_path.parent / row["audio"]).resolve()
            info = sf.info(path)
            if info.frames <= 0:
                raise ValueError(f"Empty audio: {path}")
            audio_hash = digest(path)
            for mapping, key in ((groups, row["group"]), (speakers, row["speaker"])):
                if mapping.setdefault(key, split) != split:
                    raise ValueError(f"Cross-split group or speaker: {key}")
            if audio_hash in audio_hashes:
                raise ValueError(f"Duplicate audio: {utt} and {audio_hashes[audio_hash]}")
            audio_hashes[audio_hash] = utt
            row.update(audio=str(path), sha256=audio_hash, duration_s=info.duration)
            splits[split].append(row)
    if not splits["train"] or not splits["val"]:
        raise ValueError("Both train and val must contain utterances")

    def write(name, rows):
        with (staging / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[name] = {
            lang: round(sum(row["duration_s"] for row in rows if row["lang"] == lang), 3)
            for lang in QUOTAS
        }

    def order_key(row):
        return hashlib.sha256(f"{seed}:{row['utt']}".encode()).digest()

    with new_directory(output) as staging:
        summary = {}
        for split, rows in splits.items():
            write(split, rows)
        pools = {lang: sorted(
            (row for row in splits["train"] if row["lang"] == lang),
            key=order_key
        ) for lang in QUOTAS}
        for hours in (1, 3, 10):
            selected = []
            for lang, ratio in QUOTAS.items():
                duration = 0
                for row in pools[lang]:
                    if duration >= hours * 3600 * ratio:
                        break
                    selected.append(row)
                    duration += row["duration_s"]
            write(f"train_{hours}h", sorted(selected, key=order_key))
        save_json(staging / "duration_seconds.json", summary)


def main():
    parser = argparse.ArgumentParser(description="Validate CSV and build nested bilingual manifests")
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.csv, args.output, args.seed)


if __name__ == "__main__":
    main()

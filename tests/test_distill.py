import os
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

from m5_lite.common import activate_upstream
from m5_lite.data import prepare
from m5_lite.tracks import summarize
from m5_lite.train import losses, pass_once


activate_upstream(Path(os.environ.get("M5_UPSTREAM", "third_party/M5-TTS")))
from m5.model.flavors import FLAVORS, TransformerDecoder
from m5.model.model import CSM, ModelArgs
from transformers.cache_utils import DynamicCache
from m5_lite.depth import depth_logits, prune


def tiny_model():
    FLAVORS["test-tiny"] = lambda attention: TransformerDecoder(
        num_layers=3, num_heads=4, num_kv_heads=2, embed_dim=32,
        intermediate_dim=64, attention_type=attention,
    )
    model = CSM(ModelArgs("test-tiny", "test-tiny", 128, 33, 16))
    with torch.no_grad():
        model.audio_head.normal_(std=0.1)
    return model.eval()


class DistillationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def setUp(self):
        self.model = tiny_model()
        self.hidden = torch.randn(3, 32)
        self.codes = torch.randint(1, 33, (3, 16))

    @torch.no_grad()
    def incremental_parity(self):
        model = self.model
        full = depth_logits(model, self.hidden, self.codes)
        cache = DynamicCache(config=model.decoder.config)
        for index in range(1, 16):
            current = model._embed_audio(index - 1, self.codes[:, index - 1])[:, None]
            pos = torch.tensor([[index]])
            if index == 1:
                current = torch.cat((self.hidden[:, None], current), dim=1)
                pos = torch.tensor([[0, 1]])
            decoded = model.decoder(model.projection(current), None, pos, cache, use_cache=True)
            torch.testing.assert_close(full[:, index - 1], decoded[:, -1] @ model.audio_head[index - 1],
                                       atol=2e-6, rtol=2e-5)
        return full

    def test_causal_positions_before_and_after_pruning(self):
        self.incremental_parity()
        prune(self.model, [0, 2], 32)
        self.assertEqual(len(self.model.decoder.layers), 2)
        self.assertEqual(self.model.decoder.layers[1].self_attn.layer_idx, 1)
        self.incremental_parity()

    def test_full_width_weight_migration(self):
        expected = self.incremental_parity()
        prune(self.model, [0, 1, 2], 64)
        torch.testing.assert_close(self.incremental_parity(), expected)

    def test_cached_targets_and_frozen_weights(self):
        with torch.no_grad():
            teacher_logits = depth_logits(self.model, self.hidden, self.codes)
        targets = summarize(teacher_logits, 16, 2)
        batch = targets | {"hidden": self.hidden, "codes": self.codes.to(torch.int16)}
        torch.testing.assert_close(targets["prob"].sum(-1) + targets["tail"], torch.ones(3, 15))
        kl, _ = losses(teacher_logits, batch, 2)
        self.assertLess(abs(kl.item()), 1e-6)
        prune(self.model, [0, 2], 32)
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        optimizer = torch.optim.AdamW(self.model.decoder.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(batch, str(root / "shard.safetensors"))
            metrics = pass_once(self.model, root, {"shards": [{"file": "shard.safetensors", "frames": 3}],
                                                  "frames": 3, "temperature": 2},
                                {"batch_frames": 2, "kd_weight": 0.7, "seed": 42},
                                "cpu", optimizer, epoch=1)
            self.assertGreater(metrics["loss"], 0)
            save_file(self.model.state_dict(), str(root / "weights.safetensors"))
            restored = CSM(self.model.args)
            restored.load_state_dict(load_file(str(root / "weights.safetensors")))
            torch.testing.assert_close(depth_logits(restored, self.hidden, self.codes),
                                       depth_logits(self.model, self.hidden, self.codes))
        changed = [name for name, value in self.model.named_parameters() if not torch.equal(value, before[name])]
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith("decoder.") for name in changed))

    def test_manifest_rejects_cross_split_speaker(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / "a.wav", [0.2] * 320, 16000)
            sf.write(root / "b.wav", [0.3] * 320, 16000)
            csv = root / "corpus.csv"
            csv.write_text(
                "utt,audio,text,lang,speaker,group,split,license\n"
                "a,a.wav,A,en,s1,g1,train,internal\n"
                "b,b.wav,B,en,s1,g2,val,internal\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Cross-split"):
                prepare(csv, root / "manifest", 42)
            csv.write_text(csv.read_text().replace("B,en,s1", "B,en,s2"), encoding="utf-8")
            prepare(csv, root / "manifest", 42)
            self.assertEqual((root / "manifest" / "train_1h.jsonl").read_text(),
                             (root / "manifest" / "train_10h.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()

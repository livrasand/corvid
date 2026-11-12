
"""Exports the Corvid XXS bundle to an ONNX file.

Reconstructs the Transformer in PyTorch loading weights from
models/corvid-xxs/checkpoint.json (same variable order as TF.js)
and exports it to models/corvid-xxs/model.onnx. Validates the export
comparing PyTorch and ONNX Runtime outputs.
"""

import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = ROOT / "models" / "corvid-xxs"
CHECKPOINT_PATH = BUNDLE_DIR / "checkpoint.json"
CONFIG_PATH = BUNDLE_DIR / "config.json"
OUTPUT_PATH = BUNDLE_DIR / "model.onnx"


def load_bundle():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    return config["model"], checkpoint


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["dModel"]
        ff = cfg["dFF"]
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.norm1 = nn.LayerNorm(d, eps=1e-5)
        self.norm2 = nn.LayerNorm(d, eps=1e-5)
        self.ff1 = nn.Linear(d, ff, bias=True)
        self.ff2 = nn.Linear(ff, d, bias=True)

    def forward(self, x, mask):
        b, s, d = x.shape
        h = self.norm1(x)
        # multi-head attention
        n_heads = self._n_heads
        d_head = d // n_heads
        q = self.wq(h).view(b, s, n_heads, d_head).transpose(1, 2)
        k = self.wk(h).view(b, s, n_heads, d_head).transpose(1, 2)
        v = self.wv(h).view(b, s, n_heads, d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (d_head ** 0.5)
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn @ v).transpose(1, 2).contiguous().view(b, s, d)
        out = self.wo(ctx)
        x = x + out
        h2 = self.norm2(x)
        hidden = F.gelu(self.ff1(h2), approximate="tanh")
        proj = self.ff2(hidden)
        x = x + proj
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        v = cfg["vocabSize"]
        d = cfg["dModel"]
        l = cfg["maxSeqLen"]
        n_layers = cfg["nLayers"]
        self._n_heads = cfg["nHeads"]
        self.token_emb = nn.Embedding(v, d)
        self.pos_emb = nn.Embedding(l, d)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(n_layers)]
        )
        for block in self.blocks:
            block._n_heads = cfg["nHeads"]
        self.final_norm = nn.LayerNorm(d, eps=1e-5)

    def forward(self, ids):
        b, s = ids.shape
        d = self.cfg["dModel"]
        x = self.token_emb(ids) + self.pos_emb.weight[:s].unsqueeze(0)
        mask = torch.triu(
            torch.ones(s, s, device=ids.device, dtype=torch.float32),
            diagonal=1,
        ) * -1e9
        mask = mask.view(1, 1, s, s)
        for block in self.blocks:
            x = block(x, mask)
        h = self.final_norm(x)
        logits = F.linear(h, self.token_emb.weight)
        return logits


def assign_weights(model, cfg, checkpoint):
    variables = checkpoint["variables"]
    cursor = {"i": 0}

    def take(shape):
        idx = cursor["i"]
        cursor["i"] += 1
        saved = variables[idx]
        assert list(saved["shape"]) == list(shape), (
            f"variable {idx}: shape {saved['shape']} != expected {shape}"
        )
        return torch.tensor(saved["data"], dtype=torch.float32).reshape(*shape)

    v = cfg["vocabSize"]
    d = cfg["dModel"]
    l = cfg["maxSeqLen"]
    ff = cfg["dFF"]
    n_layers = cfg["nLayers"]

    sd = {}
    sd["token_emb.weight"] = take([v, d])
    sd["pos_emb.weight"] = take([l, d])

    for li in range(n_layers):
        p = f"blocks.{li}."
        sd[f"{p}wq.weight"] = take([d, d]).t().contiguous()
        sd[f"{p}wk.weight"] = take([d, d]).t().contiguous()
        sd[f"{p}wv.weight"] = take([d, d]).t().contiguous()
        sd[f"{p}wo.weight"] = take([d, d]).t().contiguous()
        sd[f"{p}norm1.weight"] = take([d])
        sd[f"{p}norm1.bias"] = take([d])
        sd[f"{p}norm2.weight"] = take([d])
        sd[f"{p}norm2.bias"] = take([d])
        sd[f"{p}ff1.weight"] = take([d, ff]).t().contiguous()
        sd[f"{p}ff1.bias"] = take([ff])
        sd[f"{p}ff2.weight"] = take([ff, d]).t().contiguous()
        sd[f"{p}ff2.bias"] = take([d])

    sd["final_norm.weight"] = take([d])
    sd["final_norm.bias"] = take([d])

    assert cursor["i"] == len(variables), (
        f"consumed {cursor['i']} of {len(variables)} variables"
    )

    model.load_state_dict(sd, strict=True)
    return model


def main():
    cfg, checkpoint = load_bundle()
    print(f"config: vocabSize={cfg['vocabSize']} maxSeqLen={cfg['maxSeqLen']} "
          f"dModel={cfg['dModel']} nHeads={cfg['nHeads']} nLayers={cfg['nLayers']} dFF={cfg['dFF']}")
    print(f"checkpoint: epoch={checkpoint['metadata']['epoch']} "
          f"valLoss={checkpoint['metadata']['valLoss']} vars={len(checkpoint['variables'])}")

    model = TransformerLM(cfg)
    assign_weights(model, cfg, checkpoint)
    model.eval()

    # Numeric validation against a PyTorch reference (internal consistency).
    ids = torch.tensor([[2, 5, 10, 3]], dtype=torch.long)
    with torch.no_grad():
        pt_out = model(ids)

    # Export.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        ids,
        str(OUTPUT_PATH),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"ONNX written at {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")

    # ONNX Runtime validation.
    sess = ort.InferenceSession(str(OUTPUT_PATH))
    ort_out = sess.run(["logits"], {"input_ids": ids.numpy().astype(np.int64)})[0]
    diff = float(np.max(np.abs(ort_out - pt_out.numpy())))
    print(f"max abs diff (PyTorch vs ONNX Runtime): {diff:.3e}")
    assert diff < 1e-4, "ONNX output does not match PyTorch"

    # Update manifest with the ONNX path.
    manifest_path = BUNDLE_DIR / "model.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.setdefault("files", {})
    manifest["files"]["onnx"] = "model.onnx"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print("Manifest updated with files.onnx")

    # Consolidate weights into a single ONNX file (without external .data).
    onnx_model = onnx.load(str(OUTPUT_PATH), load_external_data=True)
    onnx.save(onnx_model, str(OUTPUT_PATH))
    external_data = OUTPUT_PATH.with_suffix(".onnx.data")
    if external_data.exists():
        external_data.unlink()
    print(f"ONNX consolidado: {OUTPUT_PATH.stat().st_size} bytes")

    # Final validation.
    # Final validation with ONNX Runtime on the single file.
    sess2 = ort.InferenceSession(str(OUTPUT_PATH))
    ort_out2 = sess2.run(["logits"], {"input_ids": ids.numpy().astype(np.int64)})[0]
    diff2 = float(np.max(np.abs(ort_out2 - pt_out.numpy())))
    print(f"max abs diff (single-file ONNX vs PyTorch): {diff2:.3e}")
    assert diff2 < 1e-4

    onnx_model = onnx.load(str(OUTPUT_PATH))
    onnx.checker.check_model(onnx_model)
    print("onnx.checker.check_model: OK")


if __name__ == "__main__":
    main()

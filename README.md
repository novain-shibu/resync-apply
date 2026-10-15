# resync-apply

A lossless copy drafter for code "fast apply" (a model rewrites a whole file after an edit), plus a laptop simulator
that scores any drafter with the exact pass count of a real GPU run, in seconds.

## Result (one model, one dataset, one GPU - read the limits)

On 199 real fast-apply edits (Kortix/FastApply-1.5B, Kortix FastApply dataset), the drafter needs ~1.27x fewer
verification passes than tuned SuffixDecoding (Snowflake Arctic Inference, the open method built into vLLM).
On the 169 edits never used for tuning: 1.26-1.28x (bootstrap 95% CI 1.19-1.36).
On a free Kaggle T4 in a Hugging Face loop: 1.05x faster than SuffixDecoding (56.8 s vs 59.9 s, 30 edits),
and about 37x faster than normal decoding on the first 5 edits (342.8 s -> 9.2 s). Output matched greedy decoding on
every edit compared (see REPORT section 6.4).

## How it works (short)

After a mismatch, most of the cost of copy drafting is finding where to copy from next. Each pass verifies up to
8 branches as one token tree:
- **resync**: assume the model replaced, inserted or skipped one source token, and keep copying from there
- **1-token anchors**, ranked nearest after the previous copy position, plus longer n-gram anchors (n = 32..2)
- **read frontiers** in the original code and in the edit snippet
- **learned branch sizing** from an expected-copy-length table; plain single-branch passes while copying goes well

## Reproduce (laptop, no GPU, ~1 minute)

```
python reproduce.py
```
Expected: drafter 3,978 passes (all 199 edits) / 664 (first 30); copy-oracle ceiling 2,118 / 374.
`python sim/hybrid.py` predicts T4 wall-clock from the measured cost curve (tree: 662 passes, 69.5 s predicted,
70.5 s measured).

## Files

- `sim/drafter.py` - the method (same class runs in the simulator and the GPU loop)
- `sim/` - simulator and analysis scripts (`ceiling.py` oracle, `gap.py` loss breakdown, `proj.py` hardware projection)
- `traces/oracle.json` - 199 prompts + real model outputs + measured T4 cost curve
- `kaggle/` - the GPU scripts behind every wall-clock number (`gpu_race.py` = 644-pass / 56.8 s run),
  `heldout_suffix.py` = SuffixDecoding held-out pass counts (needs `arctic-inference`)

## Limits

Batch size 1 only; the tree verifies ~6.6x more tokens per pass, so heavy batching may cancel the gain.
Not measured on A100/H100 or inside vLLM. Greedy decoding only. "Matched greedy on all N tested" is not a proof of
bit-identical output (rare fp16 near-ties exist for every method).

## Cite

Novain Shibu (2026). Resync-and-Anchor Copy Drafting for Lossless Code Fast-Apply. Zenodo. DOI: <pending>

AI assistance: research code and text were produced with Claude (Anthropic) under the author's direction.

# Data

## Provenance

BehR-WM evaluation uses WebShop and TextWorld assets released with *From Word
to World: Can Large Language Models be Implicit Text-based World Models?*
([Li et al., arXiv:2512.18832](https://arxiv.org/abs/2512.18832)).

Upstream dataset: [`X1AOX1A/LLMasWorldModels`](https://huggingface.co/datasets/X1AOX1A/LLMasWorldModels)

Immutable revision:

```text
ff6ae2b924d1a49e4b89825913887f2ea96cb282
```

The downloader passes this complete revision to Hugging Face so later upstream
changes cannot silently alter the evaluation assets.

## Vendored files

The repository includes the paired 200-task agent/WM init contexts for WebShop
and TextWorld under `data/init_contexts/`.

## Download evaluation assets

```bash
# Both environments
python scripts/download_data.py

# TextWorld only
python scripts/download_data.py --env textworld
```

The TextWorld command downloads the single-step test data, the 200-task launch
manifest, and `textworld.zip`. Extraction creates 2,700 matched `.z8`, `.json`,
and `.ni` game triplets under `data/textworld/games/`; task IDs 1–200 are the
trajectory-evaluation cohort used by the bundled init contexts.

| Path | Used by |
|---|---|
| `data/llama_factory/textworld_test_173.json` | Single-step evaluation |
| `data/eval/textworld_test.json` | 200-task Real evaluation manifest |
| `data/init_contexts/textworld/` | Actor/WM initial prompts |
| `data/textworld/games/` | Real and W2R TextWorld execution |

Re-running the downloader is safe. Cached files are reused, and archives are
not re-extracted unless `--force-extract` is supplied.

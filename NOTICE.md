# Third-party code notice

`dialogue_benchmarks` is the engine under `src/`, the plugins under `plugins/`, and the
tooling under `scripts/`. All of it was written for this project **except** the
directories listed below. Those were copied ("vendored") from upstream research
repositories so the corresponding baselines can be reproduced. They keep their own
licenses, are not covered by this project's copyright, and are left out of the
source-code copy produced by `scripts/export_source_pdf.py`, which skips every
`plugins/**/vendor/**` path.

> **Public repository note:** This public release contains **none** of the third-party code listed below. The `vendor/` directories and the four upstream `training/` directories are omitted, because several upstream repositories carry no license. To run a baseline that needs them, clone the upstream repository from the table into that method's `vendor/` folder, or into `training/` for the four listed there.

## Vendored source (not included in the public release)

| Path | Upstream | License | Notes |
|---|---|---|---|
| `plugins/methods/bob/vendor/` | [songhaoyu/BoB](https://github.com/songhaoyu/BoB) (bundles a renamed `transformers==3.5.1`) | Apache-2.0 (`LICENSE.txt` included) | 4 decoding bugs fixed locally, documented in `plugins/methods/README.md` |
| `plugins/methods/hipporag/vendor/` | [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) | MIT (`LICENSE.txt` included) | vLLM-only and GritLM files removed |
| `plugins/methods/hypergraphrag/vendor/` | [LHRLAB/HyperGraphRAG](https://github.com/LHRLAB/HyperGraphRAG) | MIT (`LICENSE.txt` included) | |
| `plugins/methods/hypermem/vendor/` | [EverMind-AI/HyperMem](https://github.com/EverMind-AI/HyperMem) | Apache-2.0 (`LICENSE.txt` included) | |
| `plugins/methods/lmedr/vendor/` | [Chenrj233/LMEDR](https://github.com/Chenrj233/LMEDR) (HuggingFace `transformers` derivative) | Apache-2.0 (`LICENSE.txt` included) | |
| `plugins/methods/raptor/vendor/` | [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor) | MIT (`LICENSE.txt` included) | |
| `plugins/methods/paa/vendor/` | [hqsiswiliam/persona-adaptive-attention](https://github.com/hqsiswiliam/persona-adaptive-attention) | no repository license; individual files carry Apache-2.0 headers (HuggingFace / NVIDIA) | internal research reproduction only |
| `plugins/methods/pathrag/vendor/` | [BUPT-GAMMA/PathRAG](https://github.com/BUPT-GAMMA/PathRAG) | README states MIT, but the repository has no LICENSE file | internal research reproduction only |
| `plugins/methods/lightmem/vendor/` | [zjunlp/LightMem](https://github.com/zjunlp/LightMem) | MIT (`LICENSE.txt` included) | |
| `plugins/methods/anchormem/vendor/` | [RayNeo-AI-2025/AnchorMem](https://github.com/RayNeo-AI-2025/AnchorMem) | MIT (`LICENSE.txt` included) | |
| `plugins/methods/memalpha/vendor/` | [wangyu-ustc/Mem-alpha](https://github.com/wangyu-ustc/Mem-alpha) (agent/memory/functions), plus a renamed `transformers==4.51.3` copy (`xlibs_memalpha/`) | Mem-alpha: README claims Apache-2.0 but the repository has no LICENSE file; bundled transformers: Apache-2.0 (`LICENSE_transformers.txt`) | internal research reproduction only (Mem-alpha part) |
| `plugins/methods/rfmem/vendor/` | [Applied-Machine-Learning-Lab/ICLR2026_RF-Mem](https://github.com/Applied-Machine-Learning-Lab/ICLR2026_RF-Mem) (`rf_mem.py`) | README claims MIT, but the repository has no LICENSE file | internal research reproduction only |
| `plugins/methods/ld_agent/vendor/` | [leolee99/LD-Agent](https://github.com/leolee99/LD-Agent) (prompts + retrieval scoring) | MIT (`LICENSE.txt` included) | |
| `plugins/methods/ss_mpc/vendor/` | SS-MPC (Jang et al., 2025, arXiv 2502.16920) -- training code from the author's own repository | no LICENSE file; written by this project's lead author, included with their consent | run unmodified by `ss_mpc/train.py` |
| `plugins/methods/emmdg/vendor/` | [EricLee8/MPDRG](https://github.com/EricLee8/MPDRG) (EMMDG, ACL 2023) -- training/inference code | no LICENSE file | internal research reproduction only; imports repointed to memalpha's vendored transformers 4.51 snapshot, METEOR/CIDEr dropped from its per-epoch scoring (patches noted in place) |
| `plugins/methods/pal/vendor/` | [kylokano/PAL](https://github.com/kylokano/PAL) -- prompt templates | no LICENSE file | internal research reproduction only |
| `plugins/methods/orig/vendor/` | [ChanLiang/ORIG](https://github.com/ChanLiang/ORIG) (`decoding.py` → `orig_decoding.py`) | no LICENSE file | internal research reproduction only; only three functions taken |

## Upstream training code outside `vendor/`

These four directories hold upstream training code that was added before the
`vendor/` convention. They stay where they are so the existing training commands
keep working. `scripts/export_source_pdf.py` excludes them (`UPSTREAM_DIRS`), the
same way it excludes `vendor/`.

| Path | Upstream | License | Notes |
|---|---|---|---|
| `plugins/methods/bob/training/` | [songhaoyu/BoB](https://github.com/songhaoyu/BoB) training code; `evaluations.py` carries an MIT header (its BLEU code is copied from NLTK) | Apache-2.0 | |
| `plugins/methods/lmedr/training/` | [Chenrj233/LMEDR](https://github.com/Chenrj233/LMEDR) `train_PersonaChat.py` / `build_data_PersonaChat.py` (adapted) | Apache-2.0 | |
| `plugins/methods/orig/training/` | [ChanLiang/ORIG](https://github.com/ChanLiang/ORIG) training code, built on Microsoft DialoGPT (`gpt2_training/`, `persona_train_kl.py`) and HuggingFace/NVIDIA GPT-2 (`lsp_model/`) | the DialoGPT parts are MIT and the GPT-2 parts Apache-2.0 (file headers); ORIG itself has no LICENSE file | internal research reproduction only |
| `plugins/methods/paa/training/` | [hqsiswiliam/persona-adaptive-attention](https://github.com/hqsiswiliam/persona-adaptive-attention) training code | no repository license | internal research reproduction only |

For the entries marked "internal research reproduction only", the upstream
repository has no usable license grant. They are used to reproduce published
baselines inside this lab and must not be redistributed (for example, by publishing
this repository) until upstream licensing is clarified.

## Loaded at runtime, not vendored

| Component | Upstream | License |
|---|---|---|
| EngramaBench scorer (used by `plugins/metrics/engrama_score`) | [julianacunadc/engramabench](https://github.com/julianacunadc/engramabench) — loaded from the clone under `data/raw/engramabench/repo/` | MIT |
| Dataset-native baseline code wrapped by `groupmembench_bm25`, `dialsim_bm25`, `evermembench_llm` | each dataset's own repository, under `data/raw/<dataset>/` | see each dataset's `meta.json` `license` field |

## Datasets

No dataset content is stored in this repository. Raw files live outside git under
`DIALOGUE_BENCHMARKS_DATA/raw/<name>/`. Each dataset's license and source URL are
recorded in `plugins/datasets/<name>/meta.json`. Some of those licenses are
non-commercial (for example ComperDial, CC-BY-NC-SA 4.0) or research-only.

## Python dependencies

These are installed from PyPI, not redistributed. They are declared in
`pyproject.toml`, along with the optional extras for each baseline.

# Open-weight classification audit — 2026-11-17

The lifetime analysis uses exact OpenRouter permaslugs. A dated version, a `:free` variant and a `:batch` variant remain different usage series. "Open" means publicly downloadable weights under any license; it does not imply OSI-approved open source.

The 435 observed text-ranking identities contain 235 open, 122 closed and 78 unresolved entries. The complete evidence mapping is in [`lifetime-registry.json`](../data/openrouter/lifetime-registry.json). Unknown entries are excluded from the confirmed-open subset rather than assigned a family-level label.

## Current leaders: actual weight files verified

All ten models selected from the final complete day (2026-11-16) were checked against their exact catalog mapping, primary Hugging Face card and public Files listing. Each listed repository contains actual `.safetensors` weight shards; the audit did not download the weight files.

| Open rank | Exact ranked identity | Public weight files |
| --- | --- | --- |
| 1 | `deepseek/deepseek-v4.1-flash-20261017` | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/main) |
| 2 | `deepseek/deepseek-v4.1-flash-20260910` | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main) |
| 3 | `z-ai/glm-5.3-flash-20260826` | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/main) |
| 4 | `stepfun/step-5-preview-20261008` | [stepfun-ai/Step-5-Preview](https://huggingface.co/stepfun-ai/Step-5-Preview/tree/main) |
| 5 | `xiaomi/mimo-v2.6-flash-20260921` | [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/tree/main) |
| 6 | `nvidia/nemotron-3-ultra-550b-a55b-20260604:free` | [nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/tree/main) |
| 7 | `minimax/minimax-m3.1-flash-20261019` | [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/tree/main) |
| 8 | `moonshotai/kimi-k3.1-pro-20261023` | [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/tree/main) |
| 9 | `z-ai/glm-5.3-20260816` | [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/tree/main) |
| 10 | `nvidia/nemotron-3.5-super-vl-20261015:free` | [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/tree/main) |

## Catalog links can precede actual weight releases

Three prominent API models are excluded pending a verified public weight release:

| API model | Primary evidence | Treatment |
| --- | --- | --- |
| Hy4 | [Tencent November 4 announcement](https://www.tencent.com/tencent-launches-hy4-joining-the-ranks-of-the-worlds-leading-open-source-ai-models/) says model weights and system cards are planned in the coming days. The `tencent/Hy4` card was unavailable in this review. | Unknown / pending. Hy4-preview is a different open checkpoint. |
| Hy4 Flash | [Tencent November 16 announcement](https://www.tencent.com/tencent-launches-hy4-flash-to-power-high-speed-cost-efficient-ai-deployment-for-everyday-workflows/) describes future plans to open-source Flash. | Unknown / pending. No family inheritance. |
| ERNIE 5.5 Flash Preview | [Official release](https://ernie.baidu.com/release/ernie-5-5-flash-preview/) describes a forthcoming weight release; [Baidu’s November 12 statement](https://www.prnewswire.com/news-releases/baidu-unveils-latest-ernie-models-amid-broader-full-stack-ai-lineup-and-debuts-token-exchange-to-power-intelligent-organizations-302937959.html) schedules it by November 25. | Unknown / pending. The already-released Flash Base is different. |

Hy4 and ERNIE 5.5 Flash Preview had `hugging_face_id` values in the API catalog, yet this review could not verify those weight artifacts. This is why a nonempty catalog field is only catalog-level evidence, and current leaders receive a separate file-availability check. A later refresh must re-review pending entries rather than silently treating promised weights as already released.

## Historical classification limits

- Current and archived catalogs establish exact identity-to-weight mappings at catalog confidence. Historical missing identities were recovered from old OpenRouter pages with explicit Model weights links and targeted primary model-card checks. The entire historical pool has not received the same direct file audit as the current top ten.
- The registry describes openness at review time, not at API launch. Catalog `created` is an API listing timestamp and is never substituted for a weight-release date. An unverified `open_weights_since` stays null. Consequently the cohorts are historical usage of models confirmed open by review, not strictly cohorts that were open on first ranked appearance.
- Qwen Max/Plus, proprietary GLM variants, Liquid LFM generations and stealth aliases are not labeled open merely because nearby family members are open. Ambiguous entries stay unknown.
- Grok-2 public weights were released after its API launch. The exact relationship between the public artifact and the ranked `grok-2-1212` snapshot was not established here, so that series remains unknown.
- The historical `qwen/qwen-2-vl-7b-instruct` webpage now redirects to Qwen2.5-VL. Its classification instead cites the exact original [Qwen2-VL-7B-Instruct card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), without transferring Qwen2.5 metadata.
- [IAPS-AI’s OpenRouter open/closed archive](https://github.com/IAPS-AI/OpenRouter-OS-Rankings) helped identify targets for review. Its merging of dated versions and variants, and retroactive openness treatment, are not used in this lifetime analysis.

The resulting top ten means **top ten among confirmed-open entries in this snapshot**, not proof that no unresolved entry could later be verified open. Rankings and forecasts must retain that qualification.

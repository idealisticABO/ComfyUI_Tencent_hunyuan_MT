# ComfyUI_Tencent_hunyuan_MT

A ComfyUI custom node for local translation using Tencent Hunyuan / HY-MT models.

## Features

- Local translation inside ComfyUI
- Supports Tencent Hunyuan / HY-MT translation models
- Supports ModelScope and HuggingFace auto-download
- Supports model shard completeness check
- Supports download timeout
- No Ollama or external API server required

## Supported Models

- HY-MT1.5-1.8B-FP8, recommended
- HY-MT1.5-1.8B
- HY-MT1.5-7B-FP8
- Hunyuan-MT-7B-fp8

GGUF, GPTQ, and AngelSlim 1.25bit models are not supported by the built-in Transformers backend yet.

## Installation

Clone this repository into your ComfyUI custom_nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/idealisticABO/ComfyUI_Tencent_hunyuan_MT.git
# -*- coding: utf-8 -*-
"""
ComfyUI Tencent Hunyuan MT translation custom node

- 支持多个 Tencent Hunyuan / HY-MT 模型选项
- 移除 4bit / int4 量化逻辑
- 使用 transformers 内置加载
- 支持 ModelScope / HuggingFace 自动下载
- 支持下载超时
- 支持 safetensors 分片完整性检查
"""

import gc
import sys
import json
import shutil
import subprocess
from pathlib import Path


try:
    import folder_paths
except Exception:
    folder_paths = None


MODEL_REGISTRY = {
    "HY-MT1.5-1.8B-FP8": {
        "hf_repo": "tencent/HY-MT1.5-1.8B-FP8",
        "modelscope_repo": "Tencent-Hunyuan/HY-MT1.5-1.8B-FP8",
        "local_dir": "HY-MT1.5-1.8B-FP8",
        "loader": "transformers_fp8",
        "need_compressed_tensors": True,
        "patch_fp8_config": True,
    },
    "HY-MT1.5-1.8B": {
        "hf_repo": "tencent/HY-MT1.5-1.8B",
        "modelscope_repo": "Tencent-Hunyuan/HY-MT1.5-1.8B",
        "local_dir": "HY-MT1.5-1.8B",
        "loader": "transformers",
        "need_compressed_tensors": False,
        "patch_fp8_config": False,
    },
    "HY-MT1.5-7B-FP8": {
        "hf_repo": "tencent/HY-MT1.5-7B-FP8",
        "modelscope_repo": "Tencent-Hunyuan/HY-MT1.5-7B-FP8",
        "local_dir": "HY-MT1.5-7B-FP8",
        "loader": "transformers_fp8",
        "need_compressed_tensors": True,
        "patch_fp8_config": True,
    },
    "Hunyuan-MT-7B-fp8": {
        "hf_repo": "tencent/Hunyuan-MT-7B-fp8",
        "modelscope_repo": "Tencent-Hunyuan/Hunyuan-MT-7B-fp8",
        "local_dir": "Hunyuan-MT-7B-fp8",
        "loader": "transformers_fp8",
        "need_compressed_tensors": True,
        "patch_fp8_config": True,
    },
}


DEFAULT_MODEL_NAME = "HY-MT1.5-1.8B-FP8"
DEFAULT_DOWNLOAD_SOURCE = "ModelScope"


LANGUAGES = {
    "Chinese / 中文": "Chinese",
    "English / 英语": "English",
    "French / 法语": "French",
    "Portuguese / 葡萄牙语": "Portuguese",
    "Spanish / 西班牙语": "Spanish",
    "Japanese / 日语": "Japanese",
    "Turkish / 土耳其语": "Turkish",
    "Russian / 俄语": "Russian",
    "Arabic / 阿拉伯语": "Arabic",
    "Korean / 韩语": "Korean",
    "Thai / 泰语": "Thai",
    "Italian / 意大利语": "Italian",
    "German / 德语": "German",
    "Vietnamese / 越南语": "Vietnamese",
    "Malay / 马来语": "Malay",
    "Indonesian / 印尼语": "Indonesian",
    "Filipino / 菲律宾语": "Filipino",
    "Hindi / 印地语": "Hindi",
    "Traditional Chinese / 繁体中文": "Traditional Chinese",
    "Polish / 波兰语": "Polish",
    "Czech / 捷克语": "Czech",
    "Dutch / 荷兰语": "Dutch",
    "Khmer / 高棉语": "Khmer",
    "Burmese / 缅甸语": "Burmese",
    "Persian / 波斯语": "Persian",
    "Gujarati / 古吉拉特语": "Gujarati",
    "Urdu / 乌尔都语": "Urdu",
    "Telugu / 泰卢固语": "Telugu",
    "Marathi / 马拉地语": "Marathi",
    "Hebrew / 希伯来语": "Hebrew",
    "Bengali / 孟加拉语": "Bengali",
    "Tamil / 泰米尔语": "Tamil",
    "Ukrainian / 乌克兰语": "Ukrainian",
    "Tibetan / 藏语": "Tibetan",
    "Kazakh / 哈萨克语": "Kazakh",
    "Mongolian / 蒙古语": "Mongolian",
    "Uyghur / 维吾尔语": "Uyghur",
    "Cantonese / 粤语": "Cantonese",
}


CHINESE_TARGETS = {
    "Chinese",
    "Traditional Chinese",
    "Cantonese",
}


_model_cache = {
    "tokenizer": None,
    "model": None,
    "cache_key": None,
    "torch": None,
}


def lazy_import_dependencies():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError(
            "\n[llm_translate] 缺少 transformers / torch 等依赖，或依赖版本不兼容。\n"
            "请确认你是在 ComfyUI-Aki 自带 Python 环境中安装依赖。\n"
            f"原始错误：{repr(e)}"
        )

    return torch, AutoTokenizer, AutoModelForCausalLM


def normalize_model_name(model_name: str) -> str:
    if not model_name:
        return DEFAULT_MODEL_NAME

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"[llm_translate] Unknown model_name: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    return model_name


def check_compressed_tensors_installed(model_name: str):
    model_name = normalize_model_name(model_name)
    model_info = MODEL_REGISTRY[model_name]

    if not model_info.get("need_compressed_tensors", False):
        return

    try:
        import compressed_tensors  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            f"\n[llm_translate] 当前模型 {model_name} 需要安装 compressed-tensors。\n"
            "请关闭 ComfyUI 后执行：\n"
            "E:\\ComfyUI-aki-v1.6\\python-comfyui\\python\\python.exe -m pip install -U compressed-tensors\n"
            f"原始错误：{repr(e)}"
        )


def normalize_download_source(download_source: str) -> str:
    if not download_source:
        return "modelscope"

    value = str(download_source).strip().lower()

    if "modelscope" in value or "魔搭" in value:
        return "modelscope"

    return "huggingface"


def get_model_root() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.models_dir) / "llm_translate"

    return Path(__file__).resolve().parent / "models"


def get_model_path(model_name: str = DEFAULT_MODEL_NAME) -> Path:
    model_name = normalize_model_name(model_name)
    model_info = MODEL_REGISTRY[model_name]
    return get_model_root() / model_info["local_dir"]


def get_repo_id(download_source: str, model_name: str = DEFAULT_MODEL_NAME) -> str:
    model_name = normalize_model_name(model_name)
    source = normalize_download_source(download_source)
    model_info = MODEL_REGISTRY[model_name]

    if source == "modelscope":
        return model_info["modelscope_repo"]

    return model_info["hf_repo"]


def read_index_expected_shards(model_path: Path):
    index_path = Path(model_path) / "model.safetensors.index.json"

    if not index_path.exists():
        return []

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        shards = sorted(set(weight_map.values()))
        return shards
    except Exception as e:
        print(f"[llm_translate] Warning: failed to read index json: {repr(e)}")
        return []


def has_tokenizer_file(model_path: Path) -> bool:
    model_path = Path(model_path)

    candidates = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]

    return any((model_path / filename).exists() for filename in candidates)


def validate_model_files(model_path: Path):
    model_path = Path(model_path)
    missing = []

    if not model_path.exists():
        return False, ["<model directory>"]

    if not (model_path / "config.json").exists():
        missing.append("config.json")

    if not has_tokenizer_file(model_path):
        missing.append("tokenizer files")

    expected_shards = read_index_expected_shards(model_path)

    if expected_shards:
        for filename in expected_shards:
            path = model_path / filename
            if not path.exists():
                missing.append(filename)
            elif path.stat().st_size < 1024:
                missing.append(f"{filename} (too small)")
    else:
        shards = list(model_path.glob("*.safetensors"))
        if len(shards) == 0:
            missing.append("*.safetensors")

    return len(missing) == 0, missing


def run_snapshot_download_with_timeout(
    download_source: str,
    repo_id: str,
    local_dir: Path,
    timeout_seconds: int,
):
    source = normalize_download_source(download_source)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    child_code = r"""
import os
import sys
import shutil

source = sys.argv[1]
repo_id = sys.argv[2]
local_dir = sys.argv[3]

os.makedirs(local_dir, exist_ok=True)

if source == "modelscope":
    from modelscope import snapshot_download

    try:
        snapshot_download(repo_id, local_dir=local_dir)
    except TypeError:
        cache_dir = os.path.dirname(local_dir)
        downloaded_path = snapshot_download(repo_id, cache_dir=cache_dir)
        if os.path.abspath(downloaded_path) != os.path.abspath(local_dir):
            shutil.copytree(downloaded_path, local_dir, dirs_exist_ok=True)

else:
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.md",
            "LICENSE*",
            "License*",
            "tokenizer.*",
            "vocab.*",
            "merges.txt",
            "special_tokens_map.json",
        ],
    )
"""

    cmd = [
        sys.executable,
        "-c",
        child_code,
        source,
        repo_id,
        str(local_dir),
    ]

    print(f"[llm_translate] Download source: {source}")
    print(f"[llm_translate] Repo id: {repo_id}")
    print(f"[llm_translate] Local dir: {local_dir}")
    print(f"[llm_translate] Timeout: {timeout_seconds} seconds")

    try:
        subprocess.run(
            cmd,
            timeout=int(timeout_seconds),
            check=True,
        )

    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"[llm_translate] 模型下载超过 {timeout_seconds} 秒，已自动终止下载进程。\n"
            "如果目录中已经有残缺文件，请打开 force_redownload 后重新运行。"
        )

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"[llm_translate] 模型下载失败。\n"
            f"source={source}\n"
            f"repo_id={repo_id}\n"
            f"error={repr(e)}"
        )


def ensure_model_downloaded(
    model_name: str = DEFAULT_MODEL_NAME,
    force_redownload: bool = False,
    download_source: str = DEFAULT_DOWNLOAD_SOURCE,
    download_timeout_minutes: int = 30,
) -> Path:
    model_name = normalize_model_name(model_name)
    model_path = get_model_path(model_name)
    model_path.mkdir(parents=True, exist_ok=True)

    if force_redownload and model_path.exists():
        print(f"[llm_translate] Force redownload enabled. Removing: {model_path}")
        shutil.rmtree(model_path, ignore_errors=True)
        model_path.mkdir(parents=True, exist_ok=True)

    is_ok, missing = validate_model_files(model_path)

    if not is_ok:
        print(f"[llm_translate] Model files are incomplete for {model_name}.")
        print(f"[llm_translate] Missing files: {missing}")

        repo_id = get_repo_id(download_source, model_name)
        timeout_seconds = max(60, int(download_timeout_minutes) * 60)

        print(f"[llm_translate] Downloading {repo_id} to: {model_path}")

        run_snapshot_download_with_timeout(
            download_source=download_source,
            repo_id=repo_id,
            local_dir=model_path,
            timeout_seconds=timeout_seconds,
        )

        is_ok, missing = validate_model_files(model_path)

        if not is_ok:
            raise RuntimeError(
                "[llm_translate] 下载结束，但模型文件仍然不完整。\n"
                f"模型名称：{model_name}\n"
                f"模型目录：{model_path}\n"
                f"缺失文件：{missing}\n\n"
                "建议：\n"
                "1. 删除该目录后重新运行，或打开 force_redownload；\n"
                "2. 如果 ModelScope 仍然只下载部分分片，手动从网页下载缺失的 safetensors 分片放入该目录；\n"
                "3. 或切换 download_source 为 HuggingFace 重试。"
            )

        print("[llm_translate] Model download finished.")

    return model_path


def patch_fp8_config_if_needed(model_name: str, model_path: Path):
    model_name = normalize_model_name(model_name)
    model_info = MODEL_REGISTRY[model_name]

    if not model_info.get("patch_fp8_config", False):
        return

    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        qconfig = config.get("quantization_config", None)

        if isinstance(qconfig, dict) and "ignored_layers" in qconfig and "ignore" not in qconfig:
            qconfig["ignore"] = qconfig.pop("ignored_layers")

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            print("[llm_translate] Patched fp8 config.json: ignored_layers -> ignore")

    except Exception as e:
        print(f"[llm_translate] Warning: failed to patch fp8 config.json: {repr(e)}")


def get_torch_dtype(torch, dtype_mode: str):
    if dtype_mode == "float16":
        return torch.float16

    if dtype_mode == "bfloat16":
        return torch.bfloat16

    if dtype_mode == "float32":
        return torch.float32

    if torch.cuda.is_available():
        return torch.bfloat16

    return torch.float32


def unload_model():
    torch = _model_cache.get("torch", None)

    _model_cache["model"] = None
    _model_cache["tokenizer"] = None
    _model_cache["cache_key"] = None
    _model_cache["torch"] = None

    gc.collect()

    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    force_redownload: bool = False,
    dtype_mode: str = "auto",
    download_source: str = DEFAULT_DOWNLOAD_SOURCE,
    download_timeout_minutes: int = 30,
):
    model_name = normalize_model_name(model_name)
    torch, AutoTokenizer, AutoModelForCausalLM = lazy_import_dependencies()

    model_path = ensure_model_downloaded(
        model_name=model_name,
        force_redownload=force_redownload,
        download_source=download_source,
        download_timeout_minutes=download_timeout_minutes,
    )

    patch_fp8_config_if_needed(model_name, model_path)
    check_compressed_tensors_installed(model_name)

    model_path_str = str(model_path)
    cache_key = f"{model_name}|{model_path_str}|{dtype_mode}|{download_source}"

    if (
        _model_cache["tokenizer"] is not None
        and _model_cache["model"] is not None
        and _model_cache["cache_key"] == cache_key
        and not force_redownload
    ):
        return _model_cache["tokenizer"], _model_cache["model"], torch

    unload_model()

    torch_dtype = get_torch_dtype(torch, dtype_mode)

    print(f"[llm_translate] Loading tokenizer for {model_name} from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path_str,
        use_fast=True,
        trust_remote_code=True,
    )

    print(f"[llm_translate] Loading model: {model_name}")
    print(f"[llm_translate] Model path: {model_path}")
    print(f"[llm_translate] dtype={torch_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path_str,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model.eval()

    _model_cache["tokenizer"] = tokenizer
    _model_cache["model"] = model
    _model_cache["cache_key"] = cache_key
    _model_cache["torch"] = torch

    return tokenizer, model, torch


def contains_chinese(text: str) -> bool:
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return True

    return False


def build_prompt(source_text: str, target_language: str) -> str:
    source_text = str(source_text).strip()

    if target_language in CHINESE_TARGETS or contains_chinese(source_text):
        return f"把下面的文本翻译成{target_language}，不要额外解释。\n\n{source_text}"

    return f"Translate the following segment into {target_language}, without additional explanation.\n\n{source_text}"


def postprocess_translation(raw_text: str) -> str:
    text = str(raw_text).strip()

    for marker in [
        "<|im_end|>",
        "<|endoftext|>",
        "</s>",
    ]:
        text = text.replace(marker, "")

    return text.strip()


class HunyuanMT7BFP8Translate:
    """
    ComfyUI node:
    Input text -> Tencent Hunyuan / HY-MT model -> translated text

    类名保留 HunyuanMT7BFP8Translate，用于兼容你现有的 __init__.py。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello, how are you?",
                    },
                ),
                "model_name": (
                    list(MODEL_REGISTRY.keys()),
                    {
                        "default": DEFAULT_MODEL_NAME,
                    },
                ),
                "target_language": (
                    list(LANGUAGES.keys()),
                    {
                        "default": "Chinese / 中文",
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 8192,
                        "step": 64,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                    },
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.05,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.01,
                    },
                ),
                "dtype": (
                    ["auto", "bfloat16", "float16", "float32"],
                    {
                        "default": "auto",
                    },
                ),
                "download_source": (
                    ["ModelScope", "HuggingFace"],
                    {
                        "default": DEFAULT_DOWNLOAD_SOURCE,
                    },
                ),
                "download_timeout_minutes": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 240,
                        "step": 1,
                    },
                ),
                "force_redownload": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            },
            "optional": {
                "custom_target_language": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate"
    CATEGORY = "Tencent/Hunyuan-MT"

    def translate(
        self,
        text,
        model_name,
        target_language,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        dtype,
        download_source,
        download_timeout_minutes,
        force_redownload,
        custom_target_language="",
    ):
        if text is None or not str(text).strip():
            return ("",)

        target = custom_target_language.strip() if custom_target_language.strip() else LANGUAGES[target_language]

        tokenizer, model, torch = load_model(
            model_name=model_name,
            force_redownload=force_redownload,
            dtype_mode=dtype,
            download_source=download_source,
            download_timeout_minutes=download_timeout_minutes,
        )

        prompt = build_prompt(str(text), target)

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )

        try:
            target_device = model.get_input_embeddings().weight.device
        except Exception:
            target_device = model.device

        if hasattr(encoded, "data") and isinstance(encoded.data, dict):
            model_inputs = {
                key: value.to(target_device) if hasattr(value, "to") else value
                for key, value in encoded.data.items()
            }
            input_len = model_inputs["input_ids"].shape[-1]
        elif isinstance(encoded, dict):
            model_inputs = {
                key: value.to(target_device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            input_len = model_inputs["input_ids"].shape[-1]
        else:
            input_ids = encoded.to(target_device)
            model_inputs = {"input_ids": input_ids}
            input_len = input_ids.shape[-1]

        do_sample = float(temperature) > 0.0

        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "repetition_penalty": float(repetition_penalty),
            "do_sample": do_sample,
        }

        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                }
            )

        with torch.no_grad():
            output_ids = model.generate(
                **model_inputs,
                **generation_kwargs,
            )

        generated_ids = output_ids[0][input_len:]
        raw_output = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        translated = postprocess_translation(raw_output)

        return (translated,)

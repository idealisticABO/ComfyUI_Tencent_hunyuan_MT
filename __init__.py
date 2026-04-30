from .nodes import HunyuanMT7BFP8Translate

NODE_CLASS_MAPPINGS = {
    "TencentHunyuanMTTranslate": HunyuanMT7BFP8Translate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TencentHunyuanMTTranslate": "Tencent Hunyuan MT Translate",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
import os
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()


class Config:
    # API配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
    # 智谱AI密钥
    ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "your-api-key")
    # DASHSCOPE_API_KEY
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "your-api-key")

    # 模型配置
    # MODEL_NAME = "glm-4.5-flash"          # 智谱免费的模型
    MODEL_NAME = "qwen-max"                 # 阿里云qwen模型
    MODEL_TEMPERATURE = 0.1
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云百炼平台
    # 记忆配置
    MEMORY_TABLE_NAME = "agent_memory"
    MEMORY_MAX_ENTRIES = 1000

    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        return {
            "model": cls.MODEL_NAME,
            "temperature": cls.MODEL_TEMPERATURE,
            "api_key": cls.DASHSCOPE_API_KEY
            # "api_key": cls.ZHIPUAI_API_KEY
            # "api_key": cls.OPENAI_API_KEY
        }

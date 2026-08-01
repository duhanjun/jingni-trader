"""
LLM 客户端：调用 LLM 生成深度解读文本

skill 运行本质就是 agent 在调用 skill，因此 LLM 调用直接在 skill 内完成。

配置（通过环境变量，QUANT_ 前缀避免冲突）：
  QUANT_LLM_API_KEY:  API Key（必需）
  QUANT_LLM_BASE_URL: API 地址（默认 https://api.deepseek.com/v1）
  QUANT_LLM_MODEL:    模型名（默认 deepseek-chat）

无 API Key 时自动降级为规则模板生成（不报错，保证流程不中断）。
"""
import os
import json
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("reports-engine.llm_client")


def _get_config() -> Dict[str, str]:
    """从环境变量读取 LLM 配置"""
    return {
        "api_key": os.environ.get("QUANT_LLM_API_KEY", ""),
        "base_url": os.environ.get("QUANT_LLM_BASE_URL", "https://api.deepseek.com/v1"),
        "model": os.environ.get("QUANT_LLM_MODEL", "deepseek-chat"),
    }


def is_available() -> bool:
    """检查 LLM 是否可用（有 API Key）"""
    return bool(_get_config()["api_key"])


def call_llm(system_prompt: str, user_prompt: str,
             response_schema: Optional[Dict] = None) -> Optional[str]:
    """调用 LLM，返回原始文本响应

    参数:
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        response_schema: 期望的 JSON schema（仅用于日志，不影响调用）

    返回:
        LLM 原始文本输出；调用失败返回 None
    """
    config = _get_config()
    if not config["api_key"]:
        logger.info("未配置 QUANT_LLM_API_KEY，跳过 LLM 调用，将使用规则模板生成解读")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
        logger.info(f"调用 LLM: model={config['model']}, "
                     f"system_prompt={len(system_prompt)}字, user_prompt={len(user_prompt)}字")

        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        raw_output = response.choices[0].message.content
        logger.info(f"LLM 返回 {len(raw_output)} 字")
        return raw_output

    except ImportError:
        logger.warning("openai 库未安装，尝试用 requests 直接调用 API")
        return _call_via_requests(system_prompt, user_prompt, config)
    except Exception as e:
        logger.warning(f"LLM 调用失败: {e}")
        return None


def _call_via_requests(system_prompt: str, user_prompt: str,
                       config: Dict) -> Optional[str]:
    """无 openai 库时用 requests 直接调用"""
    try:
        import requests
        url = f"{config['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw_output = data["choices"][0]["message"]["content"]
        logger.info(f"LLM(requests) 返回 {len(raw_output)} 字")
        return raw_output
    except Exception as e:
        logger.warning(f"requests 调用 LLM 失败: {e}")
        return None


def parse_json_response(raw_output: str) -> Optional[Dict[str, Any]]:
    """从 LLM 原始输出中解析 JSON

    支持：纯 JSON / ```json ... ``` 包裹 / 前后有说明文字
    """
    if not raw_output:
        return None

    # 尝试从 markdown code block 中提取
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试直接找第一个 { 到最后一个 }
    json_start = raw_output.find('{')
    json_end = raw_output.rfind('}')
    if json_start >= 0 and json_end > json_start:
        try:
            return json.loads(raw_output[json_start:json_end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning(f"无法从 LLM 输出中解析 JSON，前200字: {raw_output[:200]}")
    return None


def generate_analysis(llm_prompt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """完整的 LLM 分析流程：调用 LLM → 解析 JSON → 返回结构化结果

    参数:
        llm_prompt: {"system_prompt": str, "user_prompt": str, "response_schema": dict}

    返回:
        解析后的 dict（如 {"trend_direction": "震荡", ...}），失败返回 None
    """
    system_prompt = llm_prompt.get("system_prompt", "")
    user_prompt = llm_prompt.get("user_prompt", "")

    if not system_prompt or not user_prompt:
        logger.warning("LLM prompt 不完整，跳过")
        return None

    raw_output = call_llm(system_prompt, user_prompt,
                          llm_prompt.get("response_schema"))
    if not raw_output:
        return None

    return parse_json_response(raw_output)

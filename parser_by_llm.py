import ollama
import json
import re

SYSTEM_PROMPT = """あなたはレシート解析AIです。OCRテキストから商品名と金額のJSONを抽出します。JSONのみ出力してください。説明不要です。"""

FEW_SHOT = """【例】
入力:
0レジ袋(小)*4
←20軽日田天領水398
2コ×単178356
1520軽#からだを想うフリー
1520軽#からだを想うAF\\128
4425軽バーガー\\358

出力:
[
  {"name": "レジ袋(小)", "price": 4},
  {"name": "日田天領水", "price": 398},
  {"name": "からだを想うフリー", "price": 356},
  {"name": "からだを想うAF", "price": 128},
  {"name": "バーガー", "price": 358}
]"""


def parse_receipt(rows: list[dict]) -> list[dict]:

    ocr_text = "\n".join(r["text"] for r in rows)

    prompt = f"""{FEW_SHOT}

【解析対象】
入力:
{ocr_text}

出力:
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        options={
            "num_ctx": 512,
            "num_predict": 512,
            "temperature": 0,  # 再現性のため0に固定
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    raw = response["message"]["content"]

    return _extract_json(raw)


def _extract_json(raw: str) -> list[dict]:
    """LLMの出力からJSONを抽出してパースする。"""

    # ```json ... ``` のコードブロックがあれば中身を取り出す
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    else:
        # コードブロックなしで [ から ] までを抽出
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            raw = m.group(0)

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []

    # 念のため型チェック
    result = []
    for item in items:
        name = item.get("name", "").strip()
        price = item.get("price")
        if name and isinstance(price, int) and price > 0:
            result.append({"name": name, "price": price})

    return result
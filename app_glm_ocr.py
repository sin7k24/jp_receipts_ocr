import base64
import json
import re
import shutil
import uuid
import os
from pathlib import Path

import requests
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "glm-ocr"

PROMPT = """\
Look at this receipt image and output a JSON array.
Rules:
- Each element must be exactly: {"name": "...", "price": 000}
- name: product name in Japanese (remove leading digits, 軽, #, arrows)
- price: integer, no yen sign
- Skip: 値引 割引 小計 合計 税 お釣り 預り 現計
- Output the JSON array only. No other text.

Output format example:
[
  {"name": "バーガー", "price": 358},
  {"name": "日田天領水", "price": 398}
]
"""

EXCLUDE_NAMES = ["値引", "割引", "小計", "合計", "税", "お釣り", "預り", "現計"]


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    input_path = TMP_DIR / f"{uid}{ext}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        items = _run_glm_ocr(str(input_path))
    finally:
        input_path.unlink(missing_ok=True)

    return {"items": items}


@app.post("/ocr/debug")
async def ocr_debug(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    input_path = TMP_DIR / f"{uid}{ext}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        with open(str(input_path), "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": PROMPT,
                "images": [image_b64],
                "stream": False,
                "options": {"num_ctx": 16384, "temperature": 0},
            },
            timeout=120,
        )
        raw = response.json().get("response", "")
    finally:
        input_path.unlink(missing_ok=True)

    return {"raw": raw}


def _run_glm_ocr(image_path: str) -> list[dict]:
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {
                "num_ctx": 16384,
                "temperature": 0,
            },
        },
        timeout=120,
    )
    response.raise_for_status()

    raw = response.json().get("response", "")
    return _extract_and_filter(raw)


def _extract_and_filter(raw: str) -> list[dict]:
    """
    GLM-OCRの出力は毎回JSON構造が異なり壊れていることも多い。
    json.loads に頼らず name/price ペアを正規表現で直接抽出する。
    """
    pairs = re.findall(
        r'"name"\s*:\s*"([^"]+)".*?"price"\s*:\s*(\d+)',
        raw,
        re.DOTALL,
    )

    result = []
    for name, price_str in pairs:
        name = name.strip()
        if not name or any(kw in name for kw in EXCLUDE_NAMES):
            continue
        price = int(price_str)
        if price <= 0:
            continue
        result.append({"name": name, "price": price})

    return result
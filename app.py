from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import uuid
import os

from ocr_engine import run_ocr
from parser import parse_receipt

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

TMP_DIR = BASE_DIR / "tmp"
OUTPUT_DIR = BASE_DIR / "output"

TMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):

    uid = str(uuid.uuid4())

    ext = os.path.splitext(file.filename)[1] or ".jpg"

    input_filename = f"{uid}{ext}"

    input_path = TMP_DIR / input_filename

    output_dir = OUTPUT_DIR / uid
    output_dir.mkdir(exist_ok=True)

    # アップロード保存
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # OCR実行 → 行情報リスト [{text, x_min, x_max, y_min, y_max, confidence}, ...]
    rows = run_ocr(
        sourceimg=str(input_path),
        output_dir=str(output_dir)
    )

    # レシート解析
    items = parse_receipt(rows)

    return {
        "items": items
    }
import sys
import json
from pathlib import Path
from types import SimpleNamespace

BASE_DIR = Path(__file__).resolve().parent

NDL_DIR = BASE_DIR / "vendor" / "ndlocr-lite" / "src"

sys.path.append(str(NDL_DIR))

import ocr


def run_ocr(sourceimg: str, output_dir: str) -> list[dict]:
    """
    OCRを実行し、各テキスト行の情報をリストで返す。

    Returns:
        list of dict:
            {
                "text": str,          # OCR認識テキスト
                "x_min": int,         # boundingBoxの左端X座標
                "x_max": int,         # boundingBoxの右端X座標
                "y_min": int,         # boundingBoxの上端Y座標
                "y_max": int,         # boundingBoxの下端Y座標
                "confidence": float,  # 認識信頼度
            }
    """
    args = SimpleNamespace(
        sourcedir=None,
        sourceimg=sourceimg,
        output=output_dir,

        viz=False,

        det_weights=str(NDL_DIR / "model" / "deim-s-1024x1024.onnx"),
        det_classes=str(NDL_DIR / "config" / "ndl.yaml"),

        det_score_threshold=0.2,
        det_conf_threshold=0.25,
        det_iou_threshold=0.2,

        simple_mode=False,

        rec_weights30=str(NDL_DIR / "model" / "parseq-ndl-24x256-30-tiny-189epoch-tegaki3-r8data-202604.onnx"),
        rec_weights50=str(NDL_DIR / "model" / "parseq-ndl-24x384-50-tiny-300epoch-tegaki3-r8data-202604.onnx"),
        rec_weights=str(NDL_DIR / "model" / "parseq-ndl-24x768-100-tiny-153epoch-tegaki3-r8data-202604.onnx"),

        rec_classes=str(NDL_DIR / "config" / "NDLmoji.yaml"),

        device="cpu",

        enable_tcy=False,

        json_only=False
    )

    ocr.process(args)

    # JSON出力を優先的に読む
    json_files = list(Path(output_dir).glob("*.json"))
    if json_files:
        return _parse_json_output(json_files[0])

    # フォールバック: テキストファイルから最低限の情報を返す
    txt_files = list(Path(output_dir).glob("*.txt"))
    if not txt_files:
        raise RuntimeError("OCR output file not found")

    with open(txt_files[0], "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    return [{"text": line, "x_min": 0, "x_max": 9999, "y_min": i * 40, "y_max": (i + 1) * 40, "confidence": 1.0}
            for i, line in enumerate(lines) if line.strip()]


def _parse_json_output(json_path: Path) -> list[dict]:
    """
    ndlocr-liteのJSON出力をパースして行情報リストに変換する。

    boundingBoxの形式:
        [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        → 左上, 左下, 右上, 右下 の順（isVertical=trueの場合）
        実際の座標範囲: x_min=min(全x), x_max=max(全x)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for block in data.get("contents", []):
        for item in block:
            text = item.get("text", "").strip()
            if not text:
                continue

            bbox = item.get("boundingBox", [])
            if not bbox:
                continue

            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]

            rows.append({
                "text": text,
                "x_min": min(xs),
                "x_max": max(xs),
                "y_min": min(ys),
                "y_max": max(ys),
                "confidence": item.get("confidence", 1.0),
            })

    # Y座標順に並び替え（上から下）
    rows.sort(key=lambda r: r["y_min"])

    return rows
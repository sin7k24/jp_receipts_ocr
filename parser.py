import re


# ---------------------------------------------------------------------------
# レシートの列構造をX座標で定義する。
#
# 実測値（img_width=1440 にスケール済みの座標）から：
#   355〜955 が全体の有効範囲
#
# 典型的なレシート列構造:
#   [行番号/記号帯]  [軽減フラグ+商品名帯]  [金額帯]
#
# 「金額帯」の開始Xを img_width に対する割合で設定する。
# 右端20%を「金額ゾーン」と見なす。
# ---------------------------------------------------------------------------

PRICE_ZONE_RATIO = 0.75   # 全行のx_maxを基準に、この割合より右側を金額ゾーンとする
MIN_PRICE = 10
MAX_PRICE = 100_000

SKIP_KEYWORDS = ["合計", "小計", "税", "ポイント", "釣り", "お預り", "レジ袋", "現計"]

# 先頭に来るノイズ（行番号、矢印、軽減税率マーク、ハッシュ）
LEADING_NOISE_RE = re.compile(r"^[\d←↑＃#]+[軽\s]*")

# 記号除去
SYMBOL_RE = re.compile(r"[＊*×xX]")

# ¥記号
YEN_RE = re.compile(r"[\\¥,]")


def parse_receipt(rows: list[dict]) -> list[dict]:
    """
    run_ocr() が返す行情報リストからレシートの商品・金額を抽出する。

    Parameters
    ----------
    rows : list of dict
        各要素は {text, x_min, x_max, y_min, y_max, confidence}

    Returns
    -------
    list of dict
        各要素は {name: str, price: int}
    """
    if not rows:
        return []

    # 画像幅を推定（全行のx_maxの最大値）
    img_x_max = max(r["x_max"] for r in rows)
    price_zone_x = img_x_max * PRICE_ZONE_RATIO

    items = []

    for row in rows:
        text = row["text"].strip()
        if not text:
            continue

        # スキップワード
        if any(kw in text for kw in SKIP_KEYWORDS):
            continue

        # ¥記号がある行はそこから金額を取得
        m = re.search(r"[\\¥](\d[\d,]+)", text)
        if m:
            price_str = YEN_RE.sub("", m.group(1))
            try:
                price = int(price_str)
            except ValueError:
                continue
            name_part = text[:m.start()]

        else:
            # ¥記号がない場合、X座標が価格ゾーンにある末尾数字を金額と判断
            #
            # ただし行全体のx_maxが価格ゾーンに届いていない行は
            # 金額列を持たない補足行（数量行など）なのでスキップ
            if row["x_max"] < price_zone_x:
                continue

            # 末尾の数字列を金額候補として抽出
            m2 = re.search(r"(\d[\d,]+)$", text)
            if not m2:
                continue

            price_str = m2.group(1).replace(",", "")
            try:
                price = int(price_str)
            except ValueError:
                continue

            name_part = text[:m2.start()]

        # 金額フィルタ
        if not (MIN_PRICE <= price <= MAX_PRICE):
            continue

        # 商品名クリーニング
        name = _clean_name(name_part)

        if len(name) < 2:
            continue

        items.append({"name": name, "price": price})

    return items


def _clean_name(raw: str) -> str:
    """商品名からOCRノイズを除去して返す。"""
    name = raw.strip()

    # 先頭の行番号・矢印・軽減税率マーク・ハッシュを除去
    name = LEADING_NOISE_RE.sub("", name)

    # 記号除去
    name = SYMBOL_RE.sub("", name)

    # ¥記号が紛れ込んでいたら除去
    name = YEN_RE.sub("", name)

    # 軽減税率マーク（単独で残る場合）
    name = name.replace("軽", "")

    return name.strip()
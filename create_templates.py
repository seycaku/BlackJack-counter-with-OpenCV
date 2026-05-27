import os
import cv2 as cv
import numpy as np

# ---------- SETTINGS ----------
SOURCE_DIR = "cards"          # папка с исходными фото карт
OUTPUT_DIR = "templates1"     # куда сохранить шаблоны

CARD_WIDTH = 200
CARD_HEIGHT = 300

CORNER_W = 36
CORNER_H = 84

RANK_OUT_W = 70
RANK_OUT_H = 125

SUIT_OUT_W = 70
SUIT_OUT_H = 100

SAVE_DEBUG = False            # True если хочешь сохранить промежуточные debug картинки
DEBUG_DIR = "template_debug"

os.makedirs(OUTPUT_DIR, exist_ok=True)
if SAVE_DEBUG:
    os.makedirs(DEBUG_DIR, exist_ok=True)

# ---------- NAME MAPS ----------
RANK_MAP = {
    "A": "Ace",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
    "10": "Ten",
    "J": "Jack",
    "Q": "Queen",
    "K": "King",
}

SUIT_MAP = {
    "S": "Spades",
    "D": "Diamonds",
    "C": "Clubs",
    "H": "Hearts",
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------- HELPERS ----------
def normalize_name(name: str) -> str:
    """
    Приводит имя файла к верхнему регистру и заменяет похожие кириллические буквы.
    Например:
    10С -> 10C
    АS -> AS
    """
    replace_map = {
        "А": "A",  # Cyrillic A
        "В": "B",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",  # Cyrillic С
        "Т": "T",
        "Х": "X",
    }
    name = name.upper().strip()
    return "".join(replace_map.get(ch, ch) for ch in name)


def parse_card_name(filename: str):
    """
    Поддерживает имена:
      AS, 10C, QH, KD ...
    Возвращает:
      ("Ace", "Spades")
    или (None, None), если имя не распознано.
    """
    stem, ext = os.path.splitext(filename)
    if ext.lower() not in VALID_EXTS:
        return None, None

    stem = normalize_name(stem)

    if len(stem) < 2:
        return None, None

    # Последний символ — масть
    suit_key = stem[-1]
    rank_key = stem[:-1]

    rank_full = RANK_MAP.get(rank_key)
    suit_full = SUIT_MAP.get(suit_key)

    if rank_full is None or suit_full is None:
        return None, None

    return rank_full, suit_full


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    center = pts.mean(axis=0)

    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]

    s = pts.sum(axis=1)
    tl_idx = np.argmin(s)
    pts = np.roll(pts, -tl_idx, axis=0)  # TL, TR, BR, BL

    return pts


def warp_card(image, contour):
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = order_points(box)

    tl, tr, br, bl = box

    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    card_w = max(int(w1), int(w2))
    card_h = max(int(h1), int(h2))

    # Если карта лежит горизонтально — поворачиваем порядок точек
    if card_w > card_h:
        box = np.array([tr, br, bl, tl], dtype="float32")

    dst = np.array([
        [0, 0],
        [CARD_WIDTH - 1, 0],
        [CARD_WIDTH - 1, CARD_HEIGHT - 1],
        [0, CARD_HEIGHT - 1]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(box, dst)
    warped = cv.warpPerspective(image, M, (CARD_WIDTH, CARD_HEIGHT))
    return warped


def corner_ink_score(region):
    gray = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return cv.countNonZero(binary)


def orient_card(warped):
    """
    Выбирает поворот, где индекс карты вероятнее всего находится в левом верхнем углу.
    """
    best_score = -1
    best = warped.copy()
    current = warped.copy()

    for _ in range(4):
        h, w = current.shape[:2]
        ph, pw = h // 5, w // 4

        tl = current[0:ph, 0:pw]
        br = current[h - ph:h, w - pw:w]
        tr = current[0:ph, w - pw:w]
        bl = current[h - ph:h, 0:pw]

        tl_score = corner_ink_score(tl)
        br_score = corner_ink_score(br)
        tr_score = corner_ink_score(tr)
        bl_score = corner_ink_score(bl)

        empty_avg = (tr_score + bl_score) / 2.0
        score = (tl_score + 0.35 * br_score) - empty_avg

        if score > best_score:
            best_score = score
            best = current.copy()

        current = cv.rotate(current, cv.ROTATE_90_CLOCKWISE)

    return best


def preprocess_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # хотим белую карту на черном фоне
    if np.mean(thresh) > 127:
        thresh = cv.bitwise_not(thresh)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel, iterations=1)

    return cleaned


def find_main_card_contour(mask, image_shape):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_h, img_w = image_shape[:2]
    img_area = img_h * img_w

    candidates = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < img_area * 0.01:
            continue
        if area > img_area * 0.95:
            continue

        rect = cv.minAreaRect(cnt)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue

        ratio = min(w, h) / max(w, h)
        if 0.50 <= ratio <= 0.82:
            candidates.append(cnt)

    if not candidates:
        return max(contours, key=cv.contourArea)

    return max(candidates, key=cv.contourArea)


def extract_corner(warped):
    corner = warped[0:CORNER_H, 0:CORNER_W]
    corner_zoom = cv.resize(corner, None, fx=4, fy=4, interpolation=cv.INTER_CUBIC)
    return corner, corner_zoom


def threshold_corner(corner_zoom):
    gray = cv.cvtColor(corner_zoom, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=1)

    return binary


def split_rank_suit(corner_binary):
    # Под твой текущий пайплайн
    rank_region = corner_binary[20:190, 10:128]
    suit_region = corner_binary[191:336, 10:128]
    return rank_region, suit_region


def center_symbol(binary, out_w, out_h):
    pts = cv.findNonZero(binary)
    canvas = np.zeros((out_h, out_w), dtype=np.uint8)

    if pts is None:
        return canvas

    x, y, w, h = cv.boundingRect(pts)
    symbol = binary[y:y + h, x:x + w]

    if symbol.size == 0:
        return canvas

    scale = min((out_w - 6) / max(w, 1), (out_h - 6) / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    symbol = cv.resize(symbol, (new_w, new_h), interpolation=cv.INTER_NEAREST)

    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2

    canvas[y0:y0 + new_h, x0:x0 + new_w] = symbol
    _, canvas = cv.threshold(canvas, 127, 255, cv.THRESH_BINARY)

    return canvas


def extract_symbol(region_binary, out_w, out_h, min_area=20):
    contours, _ = cv.findContours(region_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contours = sorted(contours, key=cv.contourArea, reverse=True)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv.boundingRect(cnt)

        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(region_binary.shape[1], x + w + pad)
        y2 = min(region_binary.shape[0], y + h + pad)

        roi = region_binary[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        return center_symbol(roi, out_w, out_h)

    return None


def process_card_image(image, debug_prefix=None):
    mask = preprocess_image(image)
    cnt = find_main_card_contour(mask, image.shape)

    if cnt is None:
        return None, None

    warped = warp_card(image, cnt)
    warped = orient_card(warped)

    corner, corner_zoom = extract_corner(warped)
    corner_binary = threshold_corner(corner_zoom)
    rank_region, suit_region = split_rank_suit(corner_binary)

    rank_symbol = extract_symbol(rank_region, RANK_OUT_W, RANK_OUT_H, min_area=25)
    suit_symbol = extract_symbol(suit_region, SUIT_OUT_W, SUIT_OUT_H, min_area=20)

    if SAVE_DEBUG and debug_prefix is not None:
        cv.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_mask.png"), mask)
        cv.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_warped.png"), warped)
        cv.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_corner_zoom.png"), corner_zoom)
        cv.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_corner_binary.png"), corner_binary)
        cv.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_rank_region.png"), rank_region)
        cv.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_suit_region.png"), suit_region)
        if rank_symbol is not None:
            cv.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_rank_symbol.png"), rank_symbol)
        if suit_symbol is not None:
            cv.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_suit_symbol.png"), suit_symbol)

    return rank_symbol, suit_symbol


def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"Папка '{SOURCE_DIR}' не найдена")
        return

    rank_done = set()
    suit_done = set()

    files = sorted(os.listdir(SOURCE_DIR))
    if not files:
        print(f"Папка '{SOURCE_DIR}' пустая")
        return

    for filename in files:
        rank_full, suit_full = parse_card_name(filename)

        if rank_full is None or suit_full is None:
            print(f"[SKIP] {filename} -> имя не распознано")
            continue

        path = os.path.join(SOURCE_DIR, filename)
        image = cv.imread(path)

        if image is None:
            print(f"[SKIP] {filename} -> не удалось прочитать")
            continue

        rank_symbol, suit_symbol = process_card_image(
            image,
            debug_prefix=os.path.splitext(filename)[0]
        )

        if rank_symbol is None:
            print(f"[WARN] {filename} -> rank не извлекся")
        if suit_symbol is None:
            print(f"[WARN] {filename} -> suit не извлекся")

        # Сохраняем rank-шаблон только один раз
        if rank_full not in rank_done and rank_symbol is not None:
            rank_path = os.path.join(OUTPUT_DIR, f"{rank_full}.jpg")
            cv.imwrite(rank_path, rank_symbol)
            rank_done.add(rank_full)
            print(f"[SAVE] rank: {rank_path}")

        # Сохраняем suit-шаблон только один раз
        if suit_full not in suit_done and suit_symbol is not None:
            suit_path = os.path.join(OUTPUT_DIR, f"{suit_full}.jpg")
            cv.imwrite(suit_path, suit_symbol)
            suit_done.add(suit_full)
            print(f"[SAVE] suit: {suit_path}")

    print("\nГотово.")
    print("Ranks:", sorted(rank_done))
    print("Suits:", sorted(suit_done))


if __name__ == "__main__":
    main()
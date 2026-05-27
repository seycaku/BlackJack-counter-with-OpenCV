import cv2 as cv
import numpy as np

CAMERA_INDEX = 2

CARD_WIDTH = 200
CARD_HEIGHT = 300

CORNER_W = 32
CORNER_H = 84

RANK_OUT_W = 70
RANK_OUT_H = 125

SUIT_OUT_W = 70
SUIT_OUT_H = 100

TEMPLATE_DIR = "templates1"

RANK_NAMES = [
    "Ace", "Two", "Three", "Four", "Five", "Six", "Seven",
    "Eight", "Nine", "Ten", "Jack", "Queen", "King"
]

SUIT_NAMES = ["Spades", "Diamonds", "Clubs", "Hearts"]

RANK_DIFF_MAX = 2400
SUIT_DIFF_MAX = 1400


def preprocess_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    if np.mean(thresh) > 127:
        thresh = cv.bitwise_not(thresh)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel, iterations=1)

    return cleaned


def find_card_contours(mask, image_shape):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    img_h, img_w = image_shape[:2]
    img_area = img_h * img_w
    possible_cards = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < img_area * 0.01 or area > img_area * 0.9:
            continue

        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) < 4 or len(approx) > 6:
            continue

        rect = cv.minAreaRect(cnt)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue

        ratio = min(w, h) / max(w, h)
        if 0.55 <= ratio <= 0.8:
            possible_cards.append(cnt)

    return sorted(possible_cards, key=cv.contourArea, reverse=True)


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    center = pts.mean(axis=0)

    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]

    tl_idx = np.argmin(pts.sum(axis=1))
    return np.roll(pts, -tl_idx, axis=0)


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

    if card_w > card_h:
        box = np.array([tr, br, bl, tl], dtype="float32")

    dst = np.array(
        [
            [0, 0],
            [CARD_WIDTH - 1, 0],
            [CARD_WIDTH - 1, CARD_HEIGHT - 1],
            [0, CARD_HEIGHT - 1],
        ],
        dtype="float32",
    )

    matrix = cv.getPerspectiveTransform(box, dst)
    return cv.warpPerspective(image, matrix, (CARD_WIDTH, CARD_HEIGHT))


def corner_ink_score(region):
    gray = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return cv.countNonZero(binary)


def orient_card(warped):
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


def extract_corner(warped):
    corner = warped[0:CORNER_H, 0:CORNER_W]
    corner_zoom = cv.resize(corner, None, fx=4, fy=4, interpolation=cv.INTER_CUBIC)
    return corner_zoom


def threshold_corner(corner_zoom):
    gray = cv.cvtColor(corner_zoom, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    return cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=1)


def split_rank_suit(corner_binary):
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

    margin = 12
    scale = min((out_w - margin) / max(w, 1), (out_h - margin) / max(h, 1))

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

    for cnt in sorted(contours, key=cv.contourArea, reverse=True):
        if cv.contourArea(cnt) < min_area:
            continue

        x, y, w, h = cv.boundingRect(cnt)
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(region_binary.shape[1], x + w + pad)
        y2 = min(region_binary.shape[0], y + h + pad)

        roi = region_binary[y1:y2, x1:x2]
        if roi.size != 0:
            return center_symbol(roi, out_w, out_h)

    return None


def preprocess_template(path, out_w, out_h):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv.GaussianBlur(img, (3, 3), 0)
    _, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=1)

    return center_symbol(binary, out_w, out_h)


def load_rank_templates():
    templates = {}

    for name in RANK_NAMES:
        path = f"{TEMPLATE_DIR}/{name}.jpg"
        tpl = preprocess_template(path, RANK_OUT_W, RANK_OUT_H)
        if tpl is not None:
            templates[name] = tpl
        else:
            print(f"Warning: cannot load rank template '{path}'")

    return templates


def load_suit_templates():
    templates = {}

    for name in SUIT_NAMES:
        path = f"{TEMPLATE_DIR}/{name}.jpg"
        tpl = preprocess_template(path, SUIT_OUT_W, SUIT_OUT_H)
        if tpl is not None:
            templates[name] = tpl
        else:
            print(f"Warning: cannot load suit template '{path}'")

    return templates


def match_template(query_symbol, templates, diff_max):
    if query_symbol is None or not templates:
        return "Unknown", 999999

    best_name = "Unknown"
    best_diff = 999999

    for name, tpl in templates.items():
        diff = cv.absdiff(query_symbol, tpl)
        diff_score = int(np.sum(diff) / 255)
        if diff_score < best_diff:
            best_diff = diff_score
            best_name = name

    if best_diff > diff_max:
        return "Unknown", best_diff

    return best_name, best_diff


def rank_to_blackjack_value(rank_name):
    if rank_name == "Ace":
        return 11
    if rank_name in ["Ten", "Jack", "Queen", "King"]:
        return 10
    if rank_name == "Two":
        return 2
    if rank_name == "Three":
        return 3
    if rank_name == "Four":
        return 4
    if rank_name == "Five":
        return 5
    if rank_name == "Six":
        return 6
    if rank_name == "Seven":
        return 7
    if rank_name == "Eight":
        return 8
    if rank_name == "Nine":
        return 9
    return 0


def blackjack_score(rank_names):
    total = 0
    aces = 0

    for rank in rank_names:
        if rank == "Unknown":
            continue

        total += rank_to_blackjack_value(rank)
        if rank == "Ace":
            aces += 1

    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    return total


def blackjack_status(total_score, card_count):
    if card_count == 0:
        return "NO CARDS"
    if total_score == 21:
        return "BLACKJACK"
    if total_score > 21:
        return "BUST"
    return "SAFE"


# def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, r=12):
#     x1, y1 = pt1
#     x2, y2 = pt2

#     if thickness < 0:
#         cv.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
#         cv.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
#         cv.circle(img, (x1 + r, y1 + r), r, color, -1)
#         cv.circle(img, (x2 - r, y1 + r), r, color, -1)
#         cv.circle(img, (x1 + r, y2 - r), r, color, -1)
#         cv.circle(img, (x2 - r, y2 - r), r, color, -1)
#     else:
#         cv.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
#         cv.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
#         cv.circle(img, (x1 + r, y1 + r), r, color, thickness)
#         cv.circle(img, (x2 - r, y1 + r), r, color, thickness)
#         cv.circle(img, (x1 + r, y2 - r), r, color, thickness)
#         cv.circle(img, (x2 - r, y2 - r), r, color, thickness)


def put_text_shadow(img, text, org, font, scale, color, thickness=1):
    cv.putText(img, text, org, font, scale, (0, 0, 0), thickness + 2, cv.LINE_AA)
    cv.putText(img, text, org, font, scale, color, thickness, cv.LINE_AA)


# def suit_display_color(suit_name):
#     if suit_name in ["Hearts", "Diamonds"]:
#         return (60, 80, 255)
#     if suit_name in ["Spades", "Clubs"]:
#         return (255, 220, 120)
#     return (180, 180, 180)


# def status_color(status):
#     if status == "BLACKJACK":
#         return (0, 220, 0)
#     if status == "BUST":
#         return (0, 0, 255)
#     if status == "SAFE":
#         return (0, 220, 255)
#     return (180, 180, 180)


def draw_card_label(frame, cnt, rank_name, suit_name):
    moments = cv.moments(cnt)
    if moments["m00"] == 0:
        return

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    line1 = f"{rank_name} of"
    line2 = suit_name
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    color = (70, 255, 175)

    (w1, h1), _ = cv.getTextSize(line1, font, scale, 2)
    (w2, h2), _ = cv.getTextSize(line2, font, scale, 2)

    text_w = max(w1, w2)
    line_gap = 10
    total_h = h1 + h2 + line_gap

    h, w = frame.shape[:2]
    x_line1 = max(5, min(w - w1 - 5, cx - w1 // 2))
    x_line2 = max(5, min(w - w2 - 5, cx - w2 // 2))
    y_line1 = max(h1 + 5, min(h - total_h + h1 - 5, cy - total_h // 2 + h1))
    y_line2 = y_line1 + h2 + line_gap

    put_text_shadow(frame, line1, (x_line1, y_line1), font, scale, color, 2)
    put_text_shadow(frame, line2, (x_line2, y_line2), font, scale, color, 2)


def draw_bottom_hud(frame, total_score, card_count):
    h, w = frame.shape[:2]
    status = blackjack_status(total_score, card_count)
    color = (70, 255, 175)
    text = f"Score: {total_score}    {status}"

    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    (tw, th), _ = cv.getTextSize(text, font, scale, 2)

    score_text = f"Score: {total_score}"
    status_text = status
    (score_w, _), _ = cv.getTextSize(score_text, font, scale, 2)

    base_y = h - 22
    score_x = (w - tw) // 2
    status_x = score_x + score_w + 24

    put_text_shadow(frame, score_text, (score_x, base_y), font, scale, color, 2)
    put_text_shadow(frame, status_text, (status_x, base_y), font, scale, color, 2)


def process_frame(image, rank_templates, suit_templates):
    max_width = 800
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv.resize(image, (int(w * scale), int(h * scale))
        )

    cleaned = preprocess_image(image)
    card_contours = find_card_contours(cleaned, image.shape)
    result = image.copy()

    detected_ranks = []

    for cnt in card_contours:
        warped = warp_card(image, cnt)
        warped = orient_card(warped)

        corner_zoom = extract_corner(warped)
        corner_binary = threshold_corner(corner_zoom)
        rank_region, suit_region = split_rank_suit(corner_binary)

        rank_symbol = extract_symbol(rank_region, RANK_OUT_W, RANK_OUT_H, min_area=25)
        suit_symbol = extract_symbol(suit_region, SUIT_OUT_W, SUIT_OUT_H, min_area=20)

        rank_name, _ = match_template(rank_symbol, rank_templates, RANK_DIFF_MAX)
        suit_name, _ = match_template(suit_symbol, suit_templates, SUIT_DIFF_MAX)

        if rank_name != "Unknown":
            detected_ranks.append(rank_name)

        cv.drawContours(result, [cnt], -1, (0, 255, 0), 2)
        draw_card_label(result, cnt, rank_name, suit_name)

    total_score = blackjack_score(detected_ranks)
    draw_bottom_hud(result, total_score, len(detected_ranks))

    return cleaned, result


def main():
    cap = cv.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: cannot open camera index {CAMERA_INDEX}")
        return

    rank_templates = load_rank_templates()
    suit_templates = load_suit_templates()

    print("Video started. Press 'q' to quit, 's' to save the current frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: cannot read frame from camera")
            break

        cleaned, result = process_frame(frame, rank_templates, suit_templates)
        cv.imshow("Detected Cards", result)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            cv.imwrite("mask_result.jpg", cleaned)
            cv.imwrite("result.jpg", result)
            print("Saved: mask_result.jpg and result.jpg")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Chuẩn hoá văn bản review tiếng Việt.

Review trên Play Store viết rất "đời": teencode (ko, dc, j), kéo dài ký tự
(quáaaa), emoji, viết hoa lung tung, dấu câu lặp. Module này đưa chúng về một
dạng thống nhất để mô hình học được tín hiệu thay vì nhiễu.
"""
from __future__ import annotations

import re
import unicodedata

# Teencode / viết tắt phổ biến trong review ví điện tử. Key là token đã lowercase.
TEENCODE = {
    "ko": "không", "k": "không", "khg": "không", "hok": "không",
    "hông": "không", "kg": "không", "khong": "không", "kô": "không", "hk": "không",
    "dc": "được", "đc": "được", "dk": "được", "đk": "được", "duoc": "được",
    "j": "gì", "gi": "gì", "z": "vậy", "v": "vậy", "vay": "vậy",
    "mn": "mọi người", "mng": "mọi người", "ng": "người", "nguoi": "người",
    "vs": "với", "voi": "với", "bt": "bình thường", "bth": "bình thường",
    "tk": "tài khoản", "sdt": "số điện thoại", "sđt": "số điện thoại",
    "nv": "nhân viên", "cskh": "chăm sóc khách hàng",
    "ad": "admin", "app": "ứng dụng", "apps": "ứng dụng", "ap": "ứng dụng",
    "lag": "lag", "lác": "lag", "lgi": "lỗi", "loi": "lỗi",
    "ok": "ok", "oke": "ok", "okie": "ok", "okela": "ok", "oki": "ok",
    "tks": "cảm ơn", "thanks": "cảm ơn", "thank": "cảm ơn", "thx": "cảm ơn", "cám": "cảm",
    "r": "rồi", "rùi": "rồi", "ròi": "rồi", "roi": "rồi",
    "wa": "quá", "qá": "quá", "qa": "quá", "wá": "quá",
    "bik": "biết", "bit": "biết", "bjt": "biết", "bít": "biết",
    "hem": "không", "hong": "không",
    "đt": "điện thoại", "dt": "điện thoại", "nhìu": "nhiều", "nhiu": "nhiều",
    "otp": "otp", "ekyc": "ekyc", "momo": "momo", "mômô": "momo",
    "good": "tốt", "nice": "tốt", "great": "tốt", "bad": "tệ", "tot": "tốt",
}

# Emoji mang cảm xúc rõ -> đổi thành token đặc biệt để giữ tín hiệu.
EMOJI_MAP = {
    "pos_emoji": "😀😃😄😁😆😊🙂😍🥰😘❤️❤💕💖💯👍👏🙏🌹😻🤩🥳✨💪👌",
    "neg_emoji": "😡😠🤬😤😞😔😢😭😩😫👎💩🙄😒😑😕☹🙁🤮😾",
}
_EMOJI_TO_TOKEN = {ch: f" {tok} " for tok, chars in EMOJI_MAP.items() for ch in chars}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PHONE_RE = re.compile(r"\b0\d{8,10}\b")
_NUM_RE = re.compile(r"\d+([.,]\d+)*\s*(k|đ|vnd|vnđ|tr|triệu|nghìn|ngàn)?\b")
_REPEAT_RE = re.compile(r"(\w)\1{2,}")  # quáaaa -> quá
_NON_WORD_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")

# Review quảng cáo app khác / mời nhập mã giới thiệu -> spam, không phản ánh trải nghiệm MoMo.
#   - mời nhập mã / tải app khác để nhận tiền
#   - tên miền bị cố tình viết lệch để lách bộ lọc: "VAY9 . C0M", "( W E B )"
#   - chiến dịch spam lặp lại (vd "TinChac")
SPAM_RE = re.compile(
    r"(nhập mã\s*\(?\s*[\d ]{6,}|mã giới thiệu\s*:?\s*\w*\d{4,}|kiếm thêm thu nhập|kiếm tiền online|"
    r"kiếm thêm (thì|lúc)|zalo:? ?0\d|tinchac|"
    r"w\s+e\s+b\b|\b[a-z]+\d*\s*\.\s*(c0m|com|vn|net|xyz)\b)",
    re.IGNORECASE,
)


def normalize(text: str | None) -> str:
    """Chuẩn hoá 1 review về dạng lowercase, đã thay teencode, không dấu câu."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    text = _URL_RE.sub(" url ", text)
    text = _PHONE_RE.sub(" phone ", text)
    for ch, tok in _EMOJI_TO_TOKEN.items():
        if ch in text:
            text = text.replace(ch, tok)
    text = _NUM_RE.sub(" num ", text)
    text = _REPEAT_RE.sub(r"\1", text)
    text = _NON_WORD_RE.sub(" ", text)
    tokens = [TEENCODE.get(t, t) for t in text.split()]
    return _SPACE_RE.sub(" ", " ".join(tokens)).strip()


def is_spam(text: str | None) -> bool:
    return isinstance(text, str) and bool(SPAM_RE.search(text))

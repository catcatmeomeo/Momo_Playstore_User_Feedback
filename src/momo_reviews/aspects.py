"""Tag reviews with product aspects using keyword lists.

The keyword lists were derived from topic modelling (NMF) on negative reviews
(see notebooks/03_topics_and_aspects.ipynb) and then refined by hand. This
rule-based approach is transparent, easy to explain to product teams and stable
over time / across releases, which a pure topic model cannot guarantee because
topics shift every time it is retrained.

Keywords stay in Vietnamese (they match review text); aspect names are English.
"""
from __future__ import annotations

import re

import pandas as pd

ASPECTS: dict[str, list[str]] = {
    "Login & OTP": ["đăng nhập", "otp", "mật khẩu", "mã xác thực", "mã xác nhận",
                        "khoá tài khoản", "khóa tài khoản", "bị khoá", "bị khóa", "không vào được"],
    "eKYC & identity": ["ekyc", "định danh", "khuôn mặt", "cccd", "căn cước", "cmnd",
                         "sinh trắc", "xác thực tài khoản", "chụp mặt", "giấy tờ"],
    "Money transfer": ["chuyển tiền", "chuyển khoản", "nhận tiền", "chuyển đi",
                           "người nhận", "qr", "quét mã"],
    "Top-up/withdraw & banks": ["nạp tiền", "rút tiền", "liên kết", "ngân hàng", "nạp vào", "rút về"],
    "Lost money & refunds": ["mất tiền", "trừ tiền", "bị trừ", "hoàn tiền", "hoàn lại",
                             "chưa nhận được tiền", "không nhận được tiền", "lừa đảo", "treo tiền"],
    "Bugs & performance": ["lỗi", "lag", "chậm", "load", "treo", "văng", "đơ", "crash",
                        "giật", "thoát ra", "không mở được", "trắng màn hình"],
    "App updates": ["cập nhật", "update", "phiên bản", "bản mới"],
    "Customer support": ["chăm sóc khách hàng", "tổng đài", "hotline", "hỗ trợ",
                            "nhân viên", "phản hồi", "khiếu nại", "trả lời"],
    "Fees": ["phí", "thu phí", "mất phí", "tính phí"],
    "Promotions & rewards": ["khuyến mãi", "ưu đãi", "voucher", "mã giảm", "giảm giá",
                            "quà", "thẻ quà", "tích điểm", "điểm thưởng"],
    "UI & ads": ["giao diện", "quảng cáo", "rối", "phức tạp", "khó dùng",
                              "khó sử dụng", "dễ sử dụng", "dễ dùng", "tiện lợi", "tiện"],
    "Loans & pay-later": ["trả sau", "khoản vay", "vay", "nợ", "tín dụng", "thu hồi nợ"],
}

_PATTERNS = {
    name: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b")
    for name, kws in ASPECTS.items()
}


def tag_aspects(text: str) -> list[str]:
    """Return the aspects mentioned in one normalised review."""
    if not text:
        return []
    return [name for name, pat in _PATTERNS.items() if pat.search(text)]


def aspect_matrix(texts: pd.Series) -> pd.DataFrame:
    """0/1 matrix with one column per aspect."""
    return pd.DataFrame({name: texts.str.contains(pat) for name, pat in _PATTERNS.items()},
                        index=texts.index).astype(int)

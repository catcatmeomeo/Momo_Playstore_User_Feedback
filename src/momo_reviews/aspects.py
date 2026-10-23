"""Gắn nhãn khía cạnh (aspect) cho review bằng bộ từ khoá.

Bộ từ khoá được xây dựng từ kết quả topic modeling (NMF) trên review tiêu cực
(xem notebooks/03_topics_and_aspects.ipynb), rồi tinh chỉnh thủ công. Cách làm
rule-based này minh bạch, dễ giải thích với team sản phẩm và theo dõi được theo
thời gian / phiên bản — điều mà topic model thuần tuý khó đảm bảo vì topic thay
đổi mỗi lần train lại.
"""
from __future__ import annotations

import re

import pandas as pd

ASPECTS: dict[str, list[str]] = {
    "Đăng nhập & OTP": ["đăng nhập", "otp", "mật khẩu", "mã xác thực", "mã xác nhận",
                        "khoá tài khoản", "khóa tài khoản", "bị khoá", "bị khóa", "không vào được"],
    "eKYC & định danh": ["ekyc", "định danh", "khuôn mặt", "cccd", "căn cước", "cmnd",
                         "sinh trắc", "xác thực tài khoản", "chụp mặt", "giấy tờ"],
    "Chuyển & nhận tiền": ["chuyển tiền", "chuyển khoản", "nhận tiền", "chuyển đi",
                           "người nhận", "qr", "quét mã"],
    "Nạp/Rút & ngân hàng": ["nạp tiền", "rút tiền", "liên kết", "ngân hàng", "nạp vào", "rút về"],
    "Mất tiền & hoàn tiền": ["mất tiền", "trừ tiền", "bị trừ", "hoàn tiền", "hoàn lại",
                             "chưa nhận được tiền", "không nhận được tiền", "lừa đảo", "treo tiền"],
    "Lỗi & hiệu năng": ["lỗi", "lag", "chậm", "load", "treo", "văng", "đơ", "crash",
                        "giật", "thoát ra", "không mở được", "trắng màn hình"],
    "Cập nhật phiên bản": ["cập nhật", "update", "phiên bản", "bản mới"],
    "Chăm sóc khách hàng": ["chăm sóc khách hàng", "tổng đài", "hotline", "hỗ trợ",
                            "nhân viên", "phản hồi", "khiếu nại", "trả lời"],
    "Phí dịch vụ": ["phí", "thu phí", "mất phí", "tính phí"],
    "Khuyến mãi & ưu đãi": ["khuyến mãi", "ưu đãi", "voucher", "mã giảm", "giảm giá",
                            "quà", "thẻ quà", "tích điểm", "điểm thưởng"],
    "Giao diện & quảng cáo": ["giao diện", "quảng cáo", "rối", "phức tạp", "khó dùng",
                              "khó sử dụng", "dễ sử dụng", "dễ dùng", "tiện lợi", "tiện"],
    "Vay & trả sau": ["trả sau", "khoản vay", "vay", "nợ", "tín dụng", "thu hồi nợ"],
}

_PATTERNS = {
    name: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b")
    for name, kws in ASPECTS.items()
}


def tag_aspects(text: str) -> list[str]:
    """Trả về danh sách aspect xuất hiện trong 1 review đã normalize."""
    if not text:
        return []
    return [name for name, pat in _PATTERNS.items() if pat.search(text)]


def aspect_matrix(texts: pd.Series) -> pd.DataFrame:
    """Ma trận 0/1: mỗi cột là 1 aspect."""
    return pd.DataFrame({name: texts.str.contains(pat) for name, pat in _PATTERNS.items()},
                        index=texts.index).astype(int)

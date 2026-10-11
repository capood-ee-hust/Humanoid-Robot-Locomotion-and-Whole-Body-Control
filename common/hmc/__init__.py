"""hmc — Humanoid Motion Control: thư viện dùng chung cho P05 và P06 (Unitree G1).

Modules
-------
robot   : nạp G1 (MuJoCo Menagerie) với actuator mô-men, CoM, Jacobian, điểm tiếp xúc
qp      : bọc OSQP với cấu trúc thưa cố định (giải lại nhanh mỗi bước)
wbc     : QP whole-body controller (floating base, tiếp xúc, ma sát, giới hạn mô-men)
safety  : state machine an toàn (STAND / DAMPING …)
sim     : góc nghiêng, cú đẩy
timing  : quỹ đạo minimum-jerk
viz     : viewer có nhịp thời gian thực, ghi GIF
"""


def ensure_utf8_output() -> None:
    """In tiếng Việt và ký hiệu (→, ✓, ≈…) an toàn trên mọi máy.

    Trên Windows, khi output bị chuyển hướng (ra file, IDE, Git Bash…), Python có thể dùng
    bảng mã cũ (cp1258/cp1252) và báo UnicodeEncodeError. Hàm này chuyển stdout/stderr sang
    UTF-8 nếu chúng chưa dùng UTF-8; trên Linux/macOS thường không làm gì.
    """
    import sys
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if enc != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


ensure_utf8_output()

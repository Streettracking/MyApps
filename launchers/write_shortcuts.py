"""Write DogRecognizeTrainer_*.lnk that open the windowless .vbs via wscript.

Run from the repo root: python3 launchers/write_shortcuts.py
Windows can also rebuild them with install_shortcuts.vbs.
"""

from __future__ import annotations

import struct
from pathlib import Path

# MS-SHLLINK header CLSID
_CLSID = bytes.fromhex("0114020000000000C000000000000046")
_HAS_LINK_INFO = 0x00000002
_HAS_RELATIVE_PATH = 0x00000008
_HAS_WORKING_DIR = 0x00000010
_HAS_ARGUMENTS = 0x00000020
_IS_UNICODE = 0x00000080


def _utf16(text: str) -> bytes:
    return struct.pack("<H", len(text)) + text.encode("utf-16-le")


def _link_info(local_ansi: str) -> bytes:
    path = local_ansi.encode("mbcs" if False else "ascii") + b"\x00"
    volume = struct.pack("<III", 17, 3, 0) + struct.pack("<I", 16) + b"\x00"
    header_size = 0x1C
    volume_off = header_size
    local_off = volume_off + len(volume)
    suffix_off = local_off + len(path)
    body = volume + path + b"\x00"
    size = header_size + len(body)
    header = struct.pack(
        "<IIIIII",
        size,
        header_size,
        1,
        volume_off,
        local_off,
        0,
    )
    # CommonPathSuffixOffset is the 6th dword; the pack above has 6 I's = 24 bytes,
    # but we need 7 fields: size, header size, flags, vol, local, network, suffix.
    header = struct.pack("<IIIII", size, header_size, 1, volume_off, local_off)
    header += struct.pack("<II", 0, suffix_off)
    if len(header) != header_size:
        raise RuntimeError("link info header %s" % len(header))
    return header + body


def write_lnk(dest: Path, vbs_name: str) -> None:
    flags = _HAS_LINK_INFO | _HAS_RELATIVE_PATH | _HAS_WORKING_DIR | _HAS_ARGUMENTS | _IS_UNICODE
    header = struct.pack("<I", 0x4C) + _CLSID
    header += struct.pack("<I", flags)
    header += struct.pack("<I", 0x20)  # archive
    header += struct.pack("<QQQ", 0, 0, 0)
    header += struct.pack("<I", 0)  # file size
    header += struct.pack("<I", 0)  # icon index
    header += struct.pack("<I", 1)  # SW_SHOWNORMAL; the vbs hides cmd itself
    header += struct.pack("<HHII", 0, 0, 0, 0)
    if len(header) != 0x4C:
        raise RuntimeError("header %s" % len(header))
    target = r"C:\Windows\System32\wscript.exe"
    info = _link_info(target)
    strings = b"".join(
        [
            _utf16(vbs_name),
            _utf16("."),
            _utf16('//B //Nologo "%s"' % vbs_name),
        ]
    )
    dest.write_bytes(header + info + strings)


def main() -> None:
    folder = Path(__file__).resolve().parent
    for name in (
        "DogRecognizeTrainer_sim",
        "DogRecognizeTrainer_robot",
        "DogRecognizeTrainer_onboard",
    ):
        write_lnk(folder / ("%s.lnk" % name), name + ".vbs")


if __name__ == "__main__":
    main()

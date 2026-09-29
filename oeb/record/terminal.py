"""Tool output as a terminal displays it.

A shell command's captured output is a BYTE STREAM written for a
terminal, not text: a progress bar prints its line, returns the cursor
to the line start (carriage return, or an ANSI cursor/erase sequence)
and prints the line again over the old one, thousands of times. The
dump keeps every one of those writes; a terminal shows only what is
left on screen, which for a long training call can be a small fraction
of the dump with every loss line still present.

render() is that terminal: it applies carriage return, backspace,
line feed and the ANSI cursor-movement / line-erase sequences, drops
colour and other display-only escapes, and returns the screen. Nothing
the screen still shows is removed or shortened — a line printed once
and never overwritten is returned byte for byte, however long the
output. A line whose content was overwritten or erased says so where it
stands (`[redrawn N times]`), so the elision is visible in the text
itself, never silent. Text with no terminal control characters is
returned unchanged (the common case: most benches).
"""
from __future__ import annotations

import re

# one token = a control sequence, or a run of printable text
_TOKEN = re.compile(
    r"(\x1b\[[0-9;?]*[ -/]*[@-~]"          # CSI: cursor, erase, colour
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC: titles, hyperlinks
    r"|\x1b[@-Z\\-_]"                       # other two-byte escapes
    r"|[\r\n\b])")
_CONTROL = re.compile(r"[\r\b\x1b]")
_OTHER_C0 = re.compile(r"[\x00-\x07\x0b-\x0c\x0e-\x1a\x1c-\x1f]")


def render(text: str) -> str:
    """The screen a terminal shows after receiving `text`."""
    if not text or not _CONTROL.search(text):
        return text
    lines: list[str] = [""]
    redrawn: list[int] = [0]   # overwrites / erasures of content per line
    armed: list[bool] = [False]  # cursor moved back over existing content
    touched: list[bool] = [False]  # any cursor/erase activity on the line
    row = col = 0

    def at(r):
        while len(lines) <= r:
            lines.append("")
            redrawn.append(0)
            armed.append(False)
            touched.append(False)

    def enter(r):
        # arriving on a line that already holds text (after a cursor-up)
        at(r)
        if lines[r]:
            armed[r] = touched[r] = True

    for tok in _TOKEN.split(text):
        if not tok:
            continue
        if tok == "\n":
            row, col = row + 1, 0
            enter(row)
        elif tok in ("\r", "\b"):
            col = 0 if tok == "\r" else max(0, col - 1)
            armed[row] = touched[row] = True
        elif tok[0] == "\x1b":
            if not tok.startswith("\x1b["):
                continue                     # OSC / two-byte: display only
            params, final = tok[2:-1], tok[-1]
            nums = [int(p) if p.isdigit() else 0
                    for p in params.lstrip("?").split(";")]
            n = nums[0] if nums and nums[0] else 1
            if final == "K":                 # erase in line
                mode = nums[0] if nums else 0
                line = lines[row]
                if mode == 1:
                    gone = line[:col + 1]
                    lines[row] = " " * len(gone) + line[col + 1:]
                elif mode == 2:
                    gone, lines[row] = line, ""
                else:
                    gone, lines[row] = line[col:], line[:col]
                if gone.strip():
                    redrawn[row] += 1
                armed[row] = False
                touched[row] = True
            elif final == "A":               # cursor up
                row = max(0, row - n)
                armed[row] = touched[row] = True
            elif final == "B":               # cursor down
                row += n
                enter(row)
            elif final == "C":
                col += n
            elif final in ("D", "G"):        # cursor left / to column
                col = max(0, col - n) if final == "D" else max(0, n - 1)
                armed[row] = touched[row] = True
            # m (colour), H/J (whole-screen moves: a captured stream has no
            # screen origin, and the scrollback keeps the text) and the rest
            # change nothing a reader of the text sees
        else:
            tok = _OTHER_C0.sub("", tok)
            if not tok:
                continue
            line = lines[row]
            if armed[row] and col < len(line):
                redrawn[row] += 1
            armed[row] = False
            if col > len(line):
                line += " " * (col - len(line))
            lines[row] = line[:col] + tok + line[col + len(tok):]
            col += len(tok)
    out = []
    for line, n, t in zip(lines, redrawn, touched):
        if t:
            line = line.rstrip(" ")
        if n:
            line += f" [redrawn {n} time{'' if n == 1 else 's'}]"
        out.append(line)
    return "\n".join(out)

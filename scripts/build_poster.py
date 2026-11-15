"""Build the poster (ROADMAP 9.2): one 16:9 slide, ``poster/poster.pptx`` (+ PDF).

A pure reader: numbers, the results table and Figure 2 come from ``docs/results/*.csv``
through :mod:`trafficrl.eval.poster`; the text is ``poster/content.yaml``; Figure 1 is drawn
with native shapes (a reduced version of the report's architecture diagram). Nothing is
simulated.

    python scripts/build_poster.py --pdf      # poster/poster.pptx, poster.pdf, figures/
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from lxml import etree  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

from trafficrl.eval import poster as P  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
W, H = 13.333, 7.5  # inches (16:9, the size of the sample poster)
FONT = "Arial"
NAVY, ACCENT, TINT, BAND = "1D1A39", "6B3FC4", "F1ECFA", "F4F3F8"
INK, MUTED, RULE, WHITE, PALE = "1B1B1F", "5B5966", "CFCBDD", "FFFFFF", "CFC8EA"
GREY, GREEN, ORANGE = "EEEEF2", "E3F1E2", "FCE8D9"  # input / safe rule / fitted on data
NO_STYLE = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # table style "No Style, No Grid"
MARKUP = re.compile(r"(\*\*.+?\*\*|~[^~]+~|\^[^^]+\^)")

# geometry (inches)
M = 0.25  # page margin
HEAD_H = 0.94
BODY_Y, STRIP_Y = 1.06, 5.86
COL_GAP = 0.22
COL_W = (3.3, 4.15, 4.943)
COL_X = (M, M + COL_W[0] + COL_GAP, M + COL_W[0] + COL_W[1] + 2 * COL_GAP)


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def add_runs(par, text: str, size: float, color: str, bold: bool = False) -> None:
    """``**bold**``, ``~subscript~`` and ``^superscript^`` -> runs of one paragraph."""
    for piece in MARKUP.split(text):
        if not piece:
            continue
        run = par.add_run()
        f = run.font
        f.name, f.size, f.bold, f.color.rgb = FONT, Pt(size), bold, rgb(color)
        if piece.startswith("**"):
            run.text, f.bold = piece[2:-2], True
        elif piece.startswith("~"):
            run.text = piece[1:-1]
            run._r.get_or_add_rPr().set("baseline", "-25000")
        elif piece.startswith("^"):
            run.text = piece[1:-1]
            run._r.get_or_add_rPr().set("baseline", "30000")
        else:
            run.text = piece


def text_box(slide, x, y, w, h, paras, name: str, anchor=MSO_ANCHOR.TOP):
    """A text box; ``paras`` = dicts with text, size and optional color / bold / align /
    before / after (pt) / bullet ("•" or "1.") / indent (in)."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for k, p in enumerate(paras):
        par = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        par.alignment = p.get("align", PP_ALIGN.LEFT)
        par.space_before, par.space_after = Pt(p.get("before", 0)), Pt(p.get("after", 0))
        par.line_spacing = p.get("spacing", 1.0)
        if p.get("bullet"):
            ppr = par._p.get_or_add_pPr()
            indent = Inches(p.get("indent", 0.13))
            ppr.set("marL", str(indent))
            ppr.set("indent", str(-indent))
            if p["bullet"] == "•":
                etree.SubElement(ppr, qn("a:buChar")).set("char", "•")
            else:
                etree.SubElement(ppr, qn("a:buAutoNum")).set("type", "arabicPeriod")
        add_runs(par, p["text"], p["size"], p.get("color", INK), p.get("bold", False))
    return box


def rect(slide, x, y, w, h, fill: str, name: str, rounded: bool = False):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.name = name
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    shp.line.fill.background()
    shp.shadow.inherit = False
    style = shp._element.find(qn("p:style"))  # theme effect reference = a drop shadow
    if style is not None:
        shp._element.remove(style)
    if rounded:
        shp.adjustments[0] = 0.08
    return shp


def box(slide, x, y, w, h, lines, fill: str, name: str, color: str = INK, edge: str = RULE):
    """A diagram box: bold first line, smaller lines below, centred."""
    shp = rect(slide, x, y, w, h, fill, name, rounded=True)
    shp.line.color.rgb = rgb(edge)
    shp.line.width = Pt(0.75)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.03)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for k, line in enumerate(lines):
        par = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        par.alignment = PP_ALIGN.CENTER
        add_runs(par, line, 7.2 if k == 0 else 6.2, color, bold=k == 0)
    return shp


def arrow(slide, x1, y1, x2, y2, name: str, head: bool = True, color: str = MUTED):
    con = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                     Inches(x2), Inches(y2))  # fmt: skip
    con.name = name
    style = con._element.find(qn("p:style"))
    if style is not None:
        con._element.remove(style)
    con.line.color.rgb = rgb(color)
    con.line.width = Pt(0.9)
    if head:
        end = etree.SubElement(con.line._get_or_add_ln(), qn("a:tailEnd"))
        end.set("type", "triangle")
        end.set("w", "med")
        end.set("len", "med")
    return con


def diagram(slide, x, y, w, t: dict) -> float:
    """Figure 1 with native shapes; returns its height (inches)."""
    n, gap = 6, 0.1
    bw = (w - (n - 1) * gap) / n
    top_h, mid_h, bot_h, vgap = 0.5, 0.62, 0.6, 0.2
    y_mid, y_bot = y + top_h + vgap, y + top_h + mid_h + 2 * vgap
    cx = [x + k * (bw + gap) for k in range(n)]
    keys = ["obs", "attention", "heads", "spread", "gate", "out"]
    for k, key in enumerate(keys):
        fill, color, edge = (GREY, INK, RULE) if key in ("obs", "out") else (TINT, INK, PALE)
        if key == "gate":
            fill, color, edge = ACCENT, WHITE, ACCENT
        box(slide, cx[k], y_mid, bw, mid_h, t[key], fill, f"Figure 1: {key}", color, edge)
        if k:
            arrow(slide, cx[k] - gap, y_mid + mid_h / 2, cx[k], y_mid + mid_h / 2,
                  f"Figure 1: arrow {k}")  # fmt: skip
    # Max-Pressure reads the same observation and feeds the gate from above
    gate_x = cx[4] + bw / 2
    mp_x, mp_w = cx[2], cx[5] + bw - cx[2] - bw - gap
    box(slide, mp_x, y, mp_w, top_h, t["mp"], GREEN, "Figure 1: max-pressure", edge="A9CDA6")
    obs_x = cx[0] + bw / 2
    arrow(slide, obs_x, y_mid, obs_x, y + top_h / 2, "Figure 1: counts up", head=False)
    arrow(slide, obs_x, y + top_h / 2, mp_x, y + top_h / 2, "Figure 1: counts to max-pressure")
    arrow(slide, gate_x, y + top_h, gate_x, y_mid, "Figure 1: max-pressure to gate")
    text_box(
        slide,
        gate_x + 0.05,
        y + top_h + 0.03,
        0.9,
        0.14,
        [{"text": t["fallback"], "size": 6, "color": MUTED}],
        "Figure 1: fallback label",
    )
    # fitted parts below: training (weights) and the threshold
    tr_x, tr_w = cx[0], cx[3] + bw - cx[0]
    box(slide, tr_x, y_bot, tr_w, bot_h, t["train"], ORANGE, "Figure 1: training", edge="EBBF9C")
    heads_x = cx[2] - gap / 2
    arrow(slide, heads_x, y_bot, heads_x, y_mid + mid_h, "Figure 1: weights")
    text_box(
        slide,
        heads_x + 0.04,
        y_mid + mid_h + 0.03,
        0.8,
        0.14,
        [{"text": t["weights"], "size": 6, "color": MUTED}],
        "Figure 1: weights label",
    )
    tau_x = cx[4]
    box(
        slide,
        tau_x,
        y_bot,
        cx[5] + bw - tau_x,
        bot_h,
        t["tau"],
        ORANGE,
        "Figure 1: threshold",
        edge="EBBF9C",
    )
    arrow(slide, gate_x, y_bot, gate_x, y_mid + mid_h, "Figure 1: threshold to gate")
    return top_h + mid_h + bot_h + 2 * vgap


def heading(text: str) -> dict:
    return {"text": text, "size": 11, "bold": True, "color": ACCENT, "after": 2}


def section(sec: dict, size: float, first: bool = True, bullet: str = "•") -> list[dict]:
    """Heading + a paragraph (``text``) or a list (``items`` / ``steps``)."""
    out = [heading(sec["heading"])]
    if not first:
        out[0]["before"] = 5
    if "text" in sec:
        out.append({"text": sec["text"], "size": size, "spacing": 1.03})
    for item in sec.get("items", []) + sec.get("steps", []):
        out.append({"text": item, "size": size, "bullet": "1." if "steps" in sec else bullet,
                    "after": 1.6, "spacing": 1.03,
                    "indent": 0.16 if "steps" in sec else 0.11})  # fmt: skip
    return out


def border(cell, side: str, color: str, width_pt: float) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    for old in tc_pr.findall(qn(f"a:{side}")):
        tc_pr.remove(old)
    ln = etree.Element(qn(f"a:{side}"), w=str(int(width_pt * 12700)), cap="flat", cmpd="sng")
    etree.SubElement(etree.SubElement(ln, qn("a:solidFill")), qn("a:srgbClr")).set("val", color)
    order = ["lnL", "lnR", "lnT", "lnB"]
    at = sum(1 for c in tc_pr if etree.QName(c).localname in order[: order.index(side)])
    tc_pr.insert(at, ln)


def add_table(slide, x, y, w, row_h, table: P.Table, widths, size: float = 7.0):
    n_rows, n_cols = len(table.rows) + 1, len(table.head)
    frame = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y), Inches(w),
                                   Inches(row_h * n_rows))  # fmt: skip
    frame.name = "Table 1: results"
    tbl = frame.table
    tbl.first_row, tbl.horz_banding = True, False
    frame._element.graphic.graphicData.tbl.tblPr.find(qn("a:tableStyleId")).text = NO_STYLE
    scale = w / sum(widths)
    for j, cw in enumerate(widths):
        tbl.columns[j].width = Inches(cw * scale)
    for i in range(n_rows):
        tbl.rows[i].height = Inches(row_h)
        cells = table.head if i == 0 else table.rows[i - 1]
        for j, value in enumerate(cells):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.03)
            cell.margin_top = cell.margin_bottom = Inches(0.012)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            par = cell.text_frame.paragraphs[0]
            par.alignment = PP_ALIGN.LEFT if j < 1 else PP_ALIGN.RIGHT
            proposed = i - 1 == table.proposed
            add_runs(par, value, size, INK,
                     bold=i == 0 or (i - 1, j) in table.bold or (proposed and j == 0))  # fmt: skip
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(TINT if proposed else WHITE)
            if i == 0:
                border(cell, "lnT", INK, 1.0)
                border(cell, "lnB", INK, 0.6)
            if i == table.first_ablation:
                border(cell, "lnB", RULE, 0.6)
            if i == n_rows - 1:
                border(cell, "lnB", INK, 1.0)
    return frame


def build(content_path: Path, results_dir: Path, out_dir: Path) -> Path:
    res = P.load(results_dir)
    c = P.fill_tree(yaml.safe_load(content_path.read_text()), P.numbers(res))
    fig_dir = out_dir / "figures"
    fig_w = COL_W[2]
    fig_h = 1.62
    fig_path = fig_dir / "robustness.png"
    plt.close(P.fig_robustness(res, fig_path, size=(fig_w, fig_h)))

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    prs.core_properties.title = c["title"]
    prs.core_properties.author = c["author"]
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    # header
    rect(slide, 0, 0, W, HEAD_H, NAVY, "Header")
    text_box(slide, M, 0.1, 8.9, HEAD_H - 0.14, [
        {"text": c["kicker"], "size": 7.5, "color": PALE, "after": 2},
        {"text": c["title"], "size": 19, "bold": True, "color": WHITE, "spacing": 0.95},
    ], "Title")  # fmt: skip
    text_box(slide, W - M - 4.3, 0.0, 4.3, HEAD_H, [
        {"text": c["author"], "size": 13, "bold": True, "color": WHITE, "align": PP_ALIGN.RIGHT,
         "after": 3},
        *({"text": line, "size": 8.5, "color": PALE, "align": PP_ALIGN.RIGHT, "after": 1}
          for line in c["author_lines"]),
    ], "Author", anchor=MSO_ANCHOR.MIDDLE)  # fmt: skip

    # left column: problem, existing frameworks, contributions, data and protocol
    body_h = STRIP_Y - BODY_Y - 0.1
    left = section(c["problem"], 7.8)
    for key in ("existing", "contributions", "setup"):
        left += section(c[key], 7.8, first=False)
    text_box(slide, COL_X[0], BODY_Y, COL_W[0], body_h, left, "Left column")

    # middle column: architecture figure, gate rule, algorithm
    x, y, w = COL_X[1], BODY_Y, COL_W[1]
    text_box(slide, x, y, w, 0.22, [heading(c["method"]["heading"])], "Method heading")
    y += 0.24
    y += diagram(slide, x, y, w, c["method"]["diagram"]) + 0.05
    text_box(slide, x, y, w, 0.14, [{"text": c["method"]["caption"], "size": 6.8,
                                     "color": MUTED}], "Figure 1 caption")  # fmt: skip
    y += 0.2
    rule_h = 0.8
    rect(slide, x, y, w, rule_h, TINT, "Gate rule background", rounded=True)
    text_box(slide, x + 0.1, y, w - 0.2, rule_h, [
        {"text": c["method"]["rule_title"], "size": 7.5, "bold": True, "color": ACCENT,
         "after": 2},
        *({"text": line, "size": 8, "after": 1.2} for line in c["method"]["rule"]),
    ], "Gate rule", anchor=MSO_ANCHOR.MIDDLE)  # fmt: skip
    y += rule_h + 0.08
    text_box(slide, x, y, w, BODY_Y + body_h - y, section(c["algorithm"], 7.6), "Algorithm 1")

    # right column: headline numbers, table, figure
    x, y, w = COL_X[2], BODY_Y, COL_W[2]
    text_box(slide, x, y, w, 0.22, [heading(c["results"]["heading"])], "Results heading")
    y += 0.25
    gap, tile_h = 0.1, 0.72
    tile_w = (w - 2 * gap) / 3
    for k, tile in enumerate(c["results"]["tiles"]):
        tx = x + k * (tile_w + gap)
        rect(slide, tx, y, tile_w, tile_h, TINT, f"Tile {k + 1} background", rounded=True)
        text_box(slide, tx + 0.08, y, tile_w - 0.16, tile_h, [
            {"text": tile["value"], "size": 16, "bold": True, "color": ACCENT},
            {"text": tile["label"], "size": 7.2, "bold": True, "after": 1},
            {"text": tile["note"], "size": 6.3, "color": MUTED},
        ], f"Tile {k + 1}", anchor=MSO_ANCHOR.MIDDLE)  # fmt: skip
    y += tile_h + 0.08
    row_h = 0.172
    table = P.results_table(res)
    add_table(slide, x, y, w, row_h, table, widths=[1.25, 0.74, 0.8, 0.8, 0.72, 0.5, 0.58])
    y += row_h * (len(table.rows) + 1) + 0.04
    text_box(slide, x, y, w, 0.22, [{"text": c["results"]["table_caption"], "size": 6.5,
                                    "color": MUTED}], "Table 1 caption")  # fmt: skip
    y += 0.27
    pic = slide.shapes.add_picture(str(fig_path), Inches(x), Inches(y), width=Inches(fig_w))
    pic.name = "Figure 2: robustness"
    y += fig_h + 0.02
    text_box(slide, x, y, w, 0.3, [{"text": c["results"]["figure_caption"], "size": 6.5,
                                    "color": MUTED}], "Figure 2 caption")  # fmt: skip

    # bottom band: findings, limitations, conclusion, references
    rect(slide, 0, STRIP_Y, W, H - STRIP_Y, BAND, "Bottom band")
    widths, sgap = (4.4, 2.8, 2.2, 2.983), 0.15
    x, y, h = M, STRIP_Y + 0.1, H - STRIP_Y - 0.2
    for key, bw in zip(("findings", "limitations", "conclusion", "references"), widths,
                       strict=True):  # fmt: skip
        if key == "references":
            paras = [heading(c[key]["heading"])] + [
                {"text": t, "size": 6.2, "color": MUTED, "after": 1.5} for t in c[key]["items"]
            ]
        else:
            paras = section(c[key], 7.5)
        text_box(slide, x, y, bw, h, paras, c[key]["heading"])
        x += bw + sgap

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "poster.pptx"
    prs.save(out)
    return out


def to_pdf(pptx: Path) -> Path:
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if exe is None:
        raise FileNotFoundError("LibreOffice (soffice) is needed for --pdf")
    subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(pptx.parent),
                    str(pptx)], check=True, capture_output=True, timeout=180)  # fmt: skip
    return pptx.with_suffix(".pdf")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--content", type=Path, default=REPO_ROOT / "poster" / "content.yaml")
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "docs" / "results")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "poster")
    ap.add_argument("--pdf", action="store_true", help="also export poster.pdf (LibreOffice)")
    args = ap.parse_args(argv)
    try:
        out = build(args.content, args.results_dir, args.out)
        print(f"wrote {out}")
        if args.pdf:
            print(f"wrote {to_pdf(out)}")
    except (FileNotFoundError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

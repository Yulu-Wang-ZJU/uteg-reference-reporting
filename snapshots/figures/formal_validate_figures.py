"""Static acceptance checks for the formal Fig. 1--4 packet."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "uteg_manuscript_figures_20261024_234843"
FIG = OUT / "figures"
DATA = OUT / "source_data"
BUILD = OUT / "build"
MAIN = ROOT / "outputs" / "uteg_manuscript_scaffold_20261008_185739" / "main.tex"
MAIN_BEFORE = BUILD / "main.tex.before"
HTML = ROOT / "outputs" / "uteg_mentor_html_20261024_220606" / "deliverables" / "index.html"
HTML_BEFORE = BUILD / "index.html.before"
SOURCE_PPT = ROOT / "outputs" / "uteg_agentic_direct_20261012_145100" / "deliverables" / "Figure1_agentic_direct.pptx"
PRIVATE_PPT = BUILD / "corrected_source_figure_final_v2.pptx"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def assert_close(actual: float, expected: float, tol: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{actual} != {expected}")


def main() -> None:
    checks: dict[str, object] = {}
    required = [
        DATA / "translation_errors.csv",
        DATA / "clinical_translation_metrics.csv",
        DATA / "RESULTS_15MIN.csv",
        DATA / "RESULTS_PRIMARY_COMPARISONS.csv",
        DATA / "mapping_source_oof.csv",
        DATA / "SEQUENTIAL.json",
        DATA / "same_5min_unreleased.csv",
        DATA / "PAIRED_EFFECTS.json",
        DATA / "figure_data_used.json",
        DATA / "fig1_corrected_source.png",
        PRIVATE_PPT,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise AssertionError("missing source or build files: " + "; ".join(missing))
    checks["required_files"] = len(required)

    stems = ["Figure1_workflow", "Figure2_translation", "Figure3_reporting", "Figure4_observation"]
    figure_files: dict[str, dict[str, object]] = {}
    for stem in stems:
        pdf, png, svg = (FIG / f"{stem}.{ext}" for ext in ("pdf", "png", "svg"))
        for path in (pdf, png, svg):
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"missing or empty figure export: {path}")
        svg_text = svg.read_text(encoding="utf-8")
        if "<svg" not in svg_text or "</svg>" not in svg_text:
            raise AssertionError(f"invalid SVG root: {svg}")
        if not pdf.read_bytes().startswith(b"%PDF"):
            raise AssertionError(f"invalid PDF signature: {pdf}")
        figure_files[stem] = {
            "pdf_bytes": pdf.stat().st_size,
            "png_bytes": png.stat().st_size,
            "svg_bytes": svg.stat().st_size,
            "png_size": png_size(png),
            "svg_has_text": "<text" in svg_text,
        }
    if figure_files["Figure1_workflow"]["png_size"] != (1600, 1280):
        raise AssertionError("Fig. 1 source export is not the expected 1600x1280 artwork")
    if "data:image/png;base64" not in (FIG / "Figure1_workflow.svg").read_text(encoding="utf-8"):
        raise AssertionError("Fig. 1 wrapper does not preserve the supplied raster artwork")
    checks["figure_exports"] = figure_files

    packet = json.loads((DATA / "figure_data_used.json").read_text(encoding="utf-8"))
    flow = packet["flow"]
    for key, row in flow.items():
        assert_close(row["correct"] + row["false_normal"] + row["withheld"], row["n"], 1e-8)
    checks["flow_totals"] = {k: round(v["correct"] + v["false_normal"] + v["withheld"], 8) for k, v in flow.items()}

    sequence = packet["sequence"]
    expected_n = {"Huzhou": 55, "SAHZU": 201, "Hubei": 41}
    by_center: dict[str, list[dict[str, object]]] = {}
    for row in sequence:
        center = row["center"]
        by_center.setdefault(center, []).append(row)
        if row["n"] != expected_n[center]:
            raise AssertionError(f"sequence denominator mismatch for {center}")
        assert_close(row["correct_yield"] + row["wrong_yield"] + row["no_report"], 1.0, 1e-9)
        assert_close(row["coverage"], row["correct_yield"] + row["wrong_yield"], 1e-9)
    for center, rows in by_center.items():
        if [r["seconds"] for r in rows] != [300, 600, 900]:
            raise AssertionError(f"unexpected CI sequence order for {center}")
        coverage = [r["coverage"] for r in rows]
        if coverage != sorted(coverage):
            raise AssertionError(f"cumulative coverage decreased for {center}")
    checks["sequence_stack_totals"] = {center: [round(r["correct_yield"] + r["wrong_yield"] + r["no_report"], 8) for r in rows] for center, rows in by_center.items()}

    paired = packet["paired_sequential"]
    if len(paired) != 6 or any(row.get("bootstrap_n") != 20000 for row in paired):
        raise AssertionError("paired CI interval packet is not six 20,000-resample rows")
    if any("ci_bonf6" not in row for row in paired):
        raise AssertionError("missing Bonferroni-6 interval")
    checks["paired_intervals"] = {"rows": len(paired), "bootstrap_n": sorted({row["bootstrap_n"] for row in paired})}

    main_text = MAIN.read_text(encoding="utf-8")
    forbidden = [
        "retained source artwork",
        "Device and media pixels",
        "saved 95\\% intervals",
        "Saved matched",
        "unified sequential branch",
        "no new resampling was performed",
    ]
    for phrase in forbidden:
        if phrase in main_text:
            raise AssertionError(f"old work-record wording remains in main.tex: {phrase}")
    expected_figures = [f"Figure{i}" for i in range(1, 5)]
    include_lines = re.findall(r"\\includegraphics\[width=\\linewidth\]\{([^}]+)\}", main_text)
    if len(include_lines) != 4 or any(stem not in include_lines[i] for i, stem in enumerate(stems)):
        raise AssertionError(f"unexpected main figure includes: {include_lines}")
    if main_text.count("\\usepackage{graphicx}") != 1:
        raise AssertionError("graphicx package declaration missing or duplicated")
    diff = list(difflib.unified_diff(MAIN_BEFORE.read_text(encoding="utf-8").splitlines(), main_text.splitlines(), fromfile=str(MAIN_BEFORE), tofile=str(MAIN), lineterm=""))
    (BUILD / "main_tex_diff.txt").write_text("\n".join(diff) + "\n", encoding="utf-8")
    checks["main_tex"] = {"before_sha256": sha256(MAIN_BEFORE), "current_sha256": sha256(MAIN), "diff_lines": len(diff), "figure_include_count": len(include_lines)}

    html_text = HTML.read_text(encoding="utf-8")
    ids = re.findall(r'id="formal-fig[1-4]"', html_text)
    if sorted(ids) != [f'id="formal-fig{i}"' for i in range(1, 5)]:
        raise AssertionError(f"formal HTML figure ids: {ids}")
    if re.search(r'<(?:script|link)[^>]+(?:src|href)=', html_text, flags=re.I):
        raise AssertionError("HTML contains an external script or stylesheet dependency")
    if html_text.count("data:image/svg+xml;base64,") != 4:
        raise AssertionError("HTML does not embed exactly four formal SVG figures")
    for legacy_id in ("maChart", "recallChart", "tradeoffChart", "ciChart"):
        if f'id="{legacy_id}"' in html_text:
            raise AssertionError(f"legacy inline chart remains in HTML: {legacy_id}")
    if "CI secondary sequence · architecture illustration" in html_text:
        raise AssertionError("legacy framework badge remains in HTML")
    checks["html"] = {"formal_figure_ids": ids, "embedded_svg_count": html_text.count("data:image/svg+xml;base64,"), "current_sha256": sha256(HTML), "before_sha256": sha256(HTML_BEFORE)}

    source_hashes = {"main_tex_before": sha256(MAIN_BEFORE), "source_ppt": sha256(SOURCE_PPT), "private_fig1_ppt": sha256(PRIVATE_PPT)}
    if source_hashes["main_tex_before"] != "F7E3624BBB0CECF59A6D0A1AAEF5CF81D3C4B6D21090C6EE873FE3A08E9DD163":
        raise AssertionError("main.tex backup hash does not match the verified pre-edit snapshot")
    if source_hashes["source_ppt"] != "CD60D69403A08E18E6D3600A829AE9AEDB7B9C641942A5AAD596AD2746CCDCB3":
        raise AssertionError("original source PPT hash changed")
    checks["untouched_source_hashes"] = source_hashes

    report = {"status": "ok", "checks": checks}
    (BUILD / "validation_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

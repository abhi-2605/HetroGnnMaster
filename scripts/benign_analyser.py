#!/usr/bin/env python3
"""
benign_operation_reporter.py

Scans the enriched TGNN graph cache, analyzes only benign samples, and produces:
1) Per-sample CSV
2) Per-operation summary CSV
3) Per-split summary CSV
4) Representative-samples CSV
5) A detailed PDF report

Usage:
    python -m scripts.benign_operation_reporter
    python -m scripts.benign_operation_reporter --split train
    python -m scripts.benign_operation_reporter --output_dir results/benign_report

Notes:
- Uses config.py for ACTIVE_GRAPH_DIR and METADATA_PATH.
- Only benign samples are analyzed.
- Works directly on .pkl HeteroData graphs.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import config
from graph_construction.pytorch_dataset import RansomwareGraphDataset

# ---------- ReportLab ----------
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ============================================================
#  CONFIG / LABEL LOGIC
# ============================================================

CLASS_BENIGN = 1

BENIGN_OP_DESCRIPTIONS = {
    "web_or_cloud_app": "Browser / cloud sync / web client",
    "network_service": "Network daemon / background service",
    "system_tool_or_installer": "System tool, updater, or installer",
    "file_operation_app": "Document editor / file manager / archiver",
    "system_configuration": "Registry / system config tool",
    "gui_application": "GUI desktop application",
    "crypto_or_security_tool": "Crypto library / security / AV helper",
    "system_monitor_or_utility": "System monitor / diagnostic / CLI utility",
    "simple_utility": "Simple utility (no strong signals)",
    "unknown_parse_error": "Could not determine (graph parse error)",
}

API_PATTERNS = {
    "network_communication": [
        "internetopen", "httpsend", "httpopen", "wsaconnect", "wsasend",
        "connect", "send", "recv", "getaddrinfo", "dnsquery",
        "internetreadfile", "internetwritefile", "wsastartup",
    ],
    "file_io": [
        "createfile", "readfile", "writefile", "deletefile", "movefile",
        "copyfile", "openfile", "setfileattributes", "getfileattributes",
        "findfile", "createdir", "removedir", "setendoffile",
    ],
    "registry_access": [
        "regopen", "regcreate", "regset", "regquery", "regdelete",
        "regclose", "regenumkey", "regenumvalue",
    ],
    "process_management": [
        "createprocess", "openprocess", "terminateprocess", "shellexecute",
        "createthread", "openthread", "suspendthread", "resumethread",
        "virtualalloc", "virtualfree", "loadlibrary", "freelibrary",
    ],
    "crypto_or_encoding": [
        "cryptgenrandom", "bcryptgenrandom", "cryptencrypt", "cryptdecrypt",
        "crypthashdata", "base64", "rtldecompressbuffer",
    ],
    "system_info": [
        "getsysteminfo", "getcomputername", "getusername", "getversion",
        "gettickcount", "querysysteminfo", "ntquerysystem",
    ],
    "ui_or_display": [
        "findwindow", "sendmessage", "postmessage", "setwindowtext",
        "createwindow", "showwindow", "messagebox",
    ],
}

BENIGN_TYPE_RULES = [
    (lambda n, e, b: b.get("network_communication", 0) > 5 and b.get("file_io", 0) > 5,
     "web_or_cloud_app"),
    (lambda n, e, b: b.get("network_communication", 0) > 5,
     "network_service"),
    (lambda n, e, b: b.get("registry_access", 0) > 5 and b.get("process_management", 0) > 3,
     "system_tool_or_installer"),
    (lambda n, e, b: b.get("file_io", 0) > 5,
     "file_operation_app"),
    (lambda n, e, b: b.get("registry_access", 0) > 5,
     "system_configuration"),
    (lambda n, e, b: b.get("ui_or_display", 0) > 3,
     "gui_application"),
    (lambda n, e, b: b.get("crypto_or_encoding", 0) > 2,
     "crypto_or_security_tool"),
    (lambda n, e, b: b.get("system_info", 0) > 3,
     "system_monitor_or_utility"),
]


def classify_api_bucket(api_names: list[str]) -> Counter:
    bucket_counts = Counter()
    for name in api_names:
        name_lower = str(name).lower()
        for bucket, keywords in API_PATTERNS.items():
            if any(kw in name_lower for kw in keywords):
                bucket_counts[bucket] += 1
                break
    return bucket_counts


def identify_benign_type(graph) -> dict:
    try:
        node_summary = {}
        for nt in graph.node_types:
            store = graph[nt]
            if hasattr(store, "x") and store.x is not None:
                node_summary[nt] = int(store.x.shape[0])

        edge_summary = {}
        for et in graph.edge_types:
            store = graph[et]
            if hasattr(store, "edge_index") and store.edge_index is not None:
                edge_summary[str(et)] = int(store.edge_index.shape[1])

        api_names = []
        if "api" in graph.node_types and hasattr(graph["api"], "api_names"):
            api_names = graph["api"].api_names or []

        bucket_hits = classify_api_bucket(api_names)

        nn = node_summary
        has_network = nn.get("network", 0) > 2
        has_file = nn.get("file", 0) > 5
        has_reg = nn.get("registry", 0) > 3

        op_type = "simple_utility"
        for cond_fn, label in BENIGN_TYPE_RULES:
            if cond_fn(nn, edge_summary, bucket_hits):
                op_type = label
                break
        else:
            if has_network and has_file:
                op_type = "web_or_cloud_app"
            elif has_network:
                op_type = "network_service"
            elif has_file and has_reg:
                op_type = "system_tool_or_installer"
            elif has_file:
                op_type = "file_operation_app"
            elif has_reg:
                op_type = "system_configuration"
            else:
                op_type = "simple_utility"

        top_apis = Counter([str(x).lower() for x in api_names]).most_common(10)

        return {
            "benign_op_type": op_type,
            "node_summary": node_summary,
            "edge_summary": edge_summary,
            "api_bucket_hits": dict(bucket_hits),
            "num_api_nodes": node_summary.get("api", 0),
            "api_names_sample": list(api_names[:10]) if api_names else [],
            "top_apis": top_apis,
        }

    except Exception as exc:
        return {
            "benign_op_type": "unknown_parse_error",
            "error": str(exc),
            "node_summary": {},
            "edge_summary": {},
            "api_bucket_hits": {},
            "num_api_nodes": 0,
            "api_names_sample": [],
            "top_apis": [],
        }


# ============================================================
#  ANALYSIS
# ============================================================

def safe_mean(values):
    return round(mean(values), 2) if values else 0.0


def safe_median(values):
    return round(median(values), 2) if values else 0.0


def safe_pct(count, total):
    return round((count / total) * 100, 2) if total else 0.0


def analyze_split(split: str) -> list[dict]:
    dataset = RansomwareGraphDataset(
        data_dir=str(config.DATA_DIR),
        split=split,
        cache_dir=str(config.ACTIVE_GRAPH_DIR),
        validate_graphs=False,
    )

    print(f"[INFO] Scanning split: {dataset.split_dir} ({len(dataset.graph_files):,} graph files)")
    rows = []

    for i, graph_file in enumerate(dataset.graph_files, 1):
        md5 = graph_file.stem
        label = dataset.labels.get(md5, 2)
        if label != CLASS_BENIGN:
            continue

        try:
            with open(graph_file, "rb") as f:
                graph = pickle.load(f)
        except Exception as exc:
            rows.append({
                "split": split,
                "md5": md5,
                "benign_op_type": "unknown_parse_error",
                "description": BENIGN_OP_DESCRIPTIONS["unknown_parse_error"],
                "api_nodes": 0,
                "file_nodes": 0,
                "registry_nodes": 0,
                "process_nodes": 0,
                "network_nodes": 0,
                "temporal_edges": 0,
                "top_bucket": "",
                "top_bucket_count": 0,
                "top_api_1": "",
                "top_api_1_count": 0,
                "top_api_2": "",
                "top_api_2_count": 0,
                "top_api_3": "",
                "top_api_3_count": 0,
                "error": str(exc),
            })
            continue

        info = identify_benign_type(graph)
        node_summary = info["node_summary"]
        bucket_hits = Counter(info["api_bucket_hits"])
        top_bucket = bucket_hits.most_common(1)[0] if bucket_hits else ("", 0)
        top_apis = info["top_apis"]

        temporal_edges = 0
        temporal_type = ("api", "temporal", "api")
        if temporal_type in graph.edge_types and hasattr(graph[temporal_type], "edge_index"):
            temporal_edges = int(graph[temporal_type].edge_index.shape[1])

        def top_api_pair(idx):
            if idx < len(top_apis):
                return top_apis[idx][0], top_apis[idx][1]
            return "", 0

        a1, a1c = top_api_pair(0)
        a2, a2c = top_api_pair(1)
        a3, a3c = top_api_pair(2)

        rows.append({
            "split": split,
            "md5": md5,
            "benign_op_type": info["benign_op_type"],
            "description": BENIGN_OP_DESCRIPTIONS.get(info["benign_op_type"], info["benign_op_type"]),
            "api_nodes": int(node_summary.get("api", 0)),
            "file_nodes": int(node_summary.get("file", 0)),
            "registry_nodes": int(node_summary.get("registry", 0)),
            "process_nodes": int(node_summary.get("process", 0)),
            "network_nodes": int(node_summary.get("network", 0)),
            "temporal_edges": temporal_edges,
            "top_bucket": top_bucket[0],
            "top_bucket_count": int(top_bucket[1]),
            "top_api_1": a1,
            "top_api_1_count": int(a1c),
            "top_api_2": a2,
            "top_api_2_count": int(a2c),
            "top_api_3": a3,
            "top_api_3_count": int(a3c),
            "error": "",
        })

        if i % 250 == 0:
            print(f"  ... processed {i}/{len(dataset.graph_files)}")

    return rows


# ============================================================
#  AGGREGATION
# ============================================================

def build_operation_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["benign_op_type"]].append(row)

    out = []
    total = len(rows)

    for op_type, items in sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True):
        api_vals = [x["api_nodes"] for x in items]
        file_vals = [x["file_nodes"] for x in items]
        reg_vals = [x["registry_nodes"] for x in items]
        proc_vals = [x["process_nodes"] for x in items]
        net_vals = [x["network_nodes"] for x in items]
        temporal_vals = [x["temporal_edges"] for x in items]

        bucket_counter = Counter()
        api_counter = Counter()
        for x in items:
            if x["top_bucket"]:
                bucket_counter[x["top_bucket"]] += 1
            for api_key, api_count_key in [
                ("top_api_1", "top_api_1_count"),
                ("top_api_2", "top_api_2_count"),
                ("top_api_3", "top_api_3_count"),
            ]:
                api_name = x[api_key]
                api_count = x[api_count_key]
                if api_name:
                    api_counter[api_name] += api_count

        out.append({
            "benign_op_type": op_type,
            "description": BENIGN_OP_DESCRIPTIONS.get(op_type, op_type),
            "count": len(items),
            "percentage": safe_pct(len(items), total),
            "avg_api_nodes": safe_mean(api_vals),
            "median_api_nodes": safe_median(api_vals),
            "avg_file_nodes": safe_mean(file_vals),
            "avg_registry_nodes": safe_mean(reg_vals),
            "avg_process_nodes": safe_mean(proc_vals),
            "avg_network_nodes": safe_mean(net_vals),
            "avg_temporal_edges": safe_mean(temporal_vals),
            "common_bucket_1": bucket_counter.most_common(1)[0][0] if bucket_counter else "",
            "common_bucket_2": bucket_counter.most_common(2)[1][0] if len(bucket_counter) > 1 else "",
            "common_api_1": api_counter.most_common(1)[0][0] if api_counter else "",
            "common_api_2": api_counter.most_common(2)[1][0] if len(api_counter) > 1 else "",
            "common_api_3": api_counter.most_common(3)[2][0] if len(api_counter) > 2 else "",
        })
    return out


def build_split_summary(rows: list[dict]) -> list[dict]:
    split_type_counter = defaultdict(Counter)
    split_totals = Counter()

    for row in rows:
        split_type_counter[row["split"]][row["benign_op_type"]] += 1
        split_totals[row["split"]] += 1

    out = []
    for split in ["train", "val", "test"]:
        for op_type, count in split_type_counter[split].most_common():
            out.append({
                "split": split,
                "benign_op_type": op_type,
                "description": BENIGN_OP_DESCRIPTIONS.get(op_type, op_type),
                "count": count,
                "percentage": safe_pct(count, split_totals[split]),
            })
    return out


def build_representative_samples(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["benign_op_type"]].append(row)

    out = []
    for op_type, items in grouped.items():
        api_vals = sorted([x["api_nodes"] for x in items])
        med = median(api_vals) if api_vals else 0
        ranked = sorted(items, key=lambda x: abs(x["api_nodes"] - med))
        for sample in ranked[:5]:
            out.append({
                "benign_op_type": op_type,
                "description": BENIGN_OP_DESCRIPTIONS.get(op_type, op_type),
                "split": sample["split"],
                "md5": sample["md5"],
                "api_nodes": sample["api_nodes"],
                "file_nodes": sample["file_nodes"],
                "registry_nodes": sample["registry_nodes"],
                "process_nodes": sample["process_nodes"],
                "network_nodes": sample["network_nodes"],
                "top_bucket": sample["top_bucket"],
                "top_api_1": sample["top_api_1"],
                "top_api_2": sample["top_api_2"],
                "top_api_3": sample["top_api_3"],
            })
    return sorted(out, key=lambda x: (x["benign_op_type"], x["split"], x["md5"]))


# ============================================================
#  CSV HELPERS
# ============================================================

def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved: {path}")


# ============================================================
#  PDF REPORT
# ============================================================

def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleX",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#17324D"),
        alignment=TA_LEFT,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="SubTitleX",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#4A5D73"),
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="SectionX",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#17324D"),
        spaceBefore=10,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="BodyX",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.4,
        leading=13,
        alignment=TA_LEFT,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="SmallX",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=10.5,
        textColor=colors.HexColor("#374151"),
    ))
    return styles


def draw_header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(colors.HexColor("#D6DEE7"))
    canvas.line(1.6*cm, h-1.35*cm, w-1.6*cm, h-1.35*cm)
    canvas.line(1.6*cm, 1.15*cm, w-1.6*cm, 1.15*cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(1.6*cm, 0.72*cm, "Benign operation analysis report")
    canvas.drawRightString(w-1.6*cm, 0.72*cm, f"Page {doc.page}")
    canvas.restoreState()


def make_table(data, col_widths=None, header_bg="#EAF1F8", font_size=8.5):
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#17324D")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), font_size),
        ("LEADING", (0,0), (-1,-1), font_size + 2),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#D8E1EA")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    return tbl


def generate_pdf_report(
    pdf_path: Path,
    rows: list[dict],
    op_summary: list[dict],
    split_summary: list[dict],
    reps: list[dict],
    selected_split: str,
):
    styles = make_styles()

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.7*cm,
        rightMargin=1.7*cm,
        topMargin=1.8*cm,
        bottomMargin=1.5*cm,
        title="Benign Operation Analysis Report",
        author="OpenAI",
    )
    story = []

    total = len(rows)
    split_counts = Counter([r["split"] for r in rows])
    type_counts = Counter([r["benign_op_type"] for r in rows])

    story.append(Paragraph("Benign Operation Analysis Report", styles["TitleX"]))
    story.append(Paragraph(
        f"Graph source: {config.ACTIVE_GRAPH_SOURCE}<br/>"
        f"Graph root: {config.ACTIVE_GRAPH_DIR}<br/>"
        f"Metadata path: {config.METADATA_PATH}<br/>"
        f"Scope: benign samples only | Split selection: {selected_split}",
        styles["SubTitleX"]
    ))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Executive summary", styles["SectionX"]))
    summary_text = (
        f"A total of <b>{total}</b> benign samples were analyzed. "
        f"The most frequent benign operation class is <b>{type_counts.most_common(1)[0][0]}</b> "
        f"with <b>{type_counts.most_common(1)[0][1]}</b> samples. "
        f"Split coverage is train={split_counts.get('train', 0)}, "
        f"val={split_counts.get('val', 0)}, test={split_counts.get('test', 0)}."
    )
    story.append(Paragraph(summary_text, styles["BodyX"]))

    exec_tbl = [
        ["Metric", "Value"],
        ["Total benign samples", f"{total:,}"],
        ["Train benign samples", f"{split_counts.get('train', 0):,}"],
        ["Val benign samples", f"{split_counts.get('val', 0):,}"],
        ["Test benign samples", f"{split_counts.get('test', 0):,}"],
        ["Distinct benign operation types", str(len(type_counts))],
        ["Most common type", f"{type_counts.most_common(1)[0][0]} ({type_counts.most_common(1)[0][1]})"],
    ]
    story.append(make_table(exec_tbl, col_widths=[6.5*cm, 9.3*cm]))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Overall operation distribution", styles["SectionX"]))
    tdata = [["Operation type", "Description", "Count", "%", "Avg API", "Median API"]]
    for row in op_summary:
        tdata.append([
            row["benign_op_type"],
            row["description"],
            str(row["count"]),
            f'{row["percentage"]:.2f}',
            f'{row["avg_api_nodes"]:.2f}',
            f'{row["median_api_nodes"]:.2f}',
        ])
    story.append(make_table(tdata, col_widths=[4.0*cm, 6.3*cm, 1.5*cm, 1.2*cm, 1.6*cm, 1.7*cm], font_size=8.1))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Split-wise distribution", styles["SectionX"]))
    sdata = [["Split", "Operation type", "Count", "%"]]
    for row in split_summary:
        sdata.append([
            row["split"],
            row["benign_op_type"],
            str(row["count"]),
            f'{row["percentage"]:.2f}',
        ])
    story.append(make_table(sdata, col_widths=[2.2*cm, 8.8*cm, 2.2*cm, 2.0*cm], font_size=8.2))
    story.append(PageBreak())

    story.append(Paragraph("Operation profiles", styles["SectionX"]))
    story.append(Paragraph(
        "The table below summarizes the structural profile of each benign operation type using average node and temporal-edge counts, along with the most characteristic API bucket and common API calls.",
        styles["BodyX"]
    ))
    pdata = [[
        "Type", "Avg API", "Avg File", "Avg Reg", "Avg Proc", "Avg Net",
        "Avg Temp", "Common bucket", "Common APIs"
    ]]
    for row in op_summary:
        common_apis = ", ".join([x for x in [row["common_api_1"], row["common_api_2"], row["common_api_3"]] if x])
        pdata.append([
            row["benign_op_type"],
            f'{row["avg_api_nodes"]:.1f}',
            f'{row["avg_file_nodes"]:.1f}',
            f'{row["avg_registry_nodes"]:.1f}',
            f'{row["avg_process_nodes"]:.1f}',
            f'{row["avg_network_nodes"]:.1f}',
            f'{row["avg_temporal_edges"]:.1f}',
            row["common_bucket_1"],
            common_apis,
        ])
    story.append(make_table(
        pdata,
        col_widths=[3.3*cm, 1.35*cm, 1.35*cm, 1.35*cm, 1.35*cm, 1.35*cm, 1.45*cm, 2.6*cm, 3.0*cm],
        font_size=7.4
    ))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Representative benign samples", styles["SectionX"]))
    story.append(Paragraph(
        "For each benign operation type, five representative samples closest to the median API-node size are listed below.",
        styles["BodyX"]
    ))
    rdata = [["Type", "Split", "MD5", "API", "File", "Reg", "Proc", "Net", "Top bucket", "Top API"]]
    for row in reps[:60]:
        rdata.append([
            row["benign_op_type"],
            row["split"],
            row["md5"],
            str(row["api_nodes"]),
            str(row["file_nodes"]),
            str(row["registry_nodes"]),
            str(row["process_nodes"]),
            str(row["network_nodes"]),
            row["top_bucket"],
            row["top_api_1"],
        ])
    story.append(make_table(
        rdata,
        col_widths=[3.0*cm, 1.2*cm, 3.2*cm, 0.9*cm, 0.9*cm, 0.9*cm, 0.9*cm, 0.9*cm, 2.3*cm, 2.4*cm],
        font_size=7.2
    ))
    story.append(PageBreak())

    story.append(Paragraph("Interpretation notes", styles["SectionX"]))
    notes = [
        "The report classifies only benign samples; ransomware and other-malware samples are ignored.",
        "Operation types are heuristic labels inferred from graph structure and API-call patterns.",
        "API-run compression or behavior-aware capping in the training pipeline does not affect this report unless the analyzed .pkl graphs themselves were permanently modified beforehand.",
        "The most reliable use of this report is to document the benign behavioral coverage seen by the model during training and evaluation."
    ]
    for note in notes:
        story.append(Paragraph(f"• {note}", styles["BodyX"]))

    doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
    print(f"[INFO] Saved: {pdf_path}")


# ============================================================
#  MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--output_dir", default="benign_operation_report")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    all_rows = []
    for split in splits:
        split_rows = analyze_split(split)
        all_rows.extend(split_rows)

    op_summary = build_operation_summary(all_rows)
    split_summary = build_split_summary(all_rows)
    reps = build_representative_samples(all_rows)

    write_csv(output_dir / "benign_per_sample.csv", all_rows)
    write_csv(output_dir / "benign_operation_summary.csv", op_summary)
    write_csv(output_dir / "benign_split_summary.csv", split_summary)
    write_csv(output_dir / "benign_representative_samples.csv", reps)

    pdf_name = f"benign_operation_report_{args.split}.pdf"
    generate_pdf_report(
        pdf_path=output_dir / pdf_name,
        rows=all_rows,
        op_summary=op_summary,
        split_summary=split_summary,
        reps=reps,
        selected_split=args.split,
    )

    print("\n" + "=" * 70)
    print(f"BENIGN REPORT COMPLETE | scope={args.split} | total benign={len(all_rows)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

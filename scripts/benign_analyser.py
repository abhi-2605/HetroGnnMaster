# #!/usr/bin/env python3
# """
# benign_operation_reporter.py

# Scans the enriched TGNN graph cache, analyzes only benign samples, and produces:
# 1) Per-sample CSV
# 2) Per-operation summary CSV
# 3) Per-split summary CSV
# 4) Representative-samples CSV
# 5) A detailed PDF report

# Usage:
#     python -m scripts.benign_operation_reporter
#     python -m scripts.benign_operation_reporter --split train
#     python -m scripts.benign_operation_reporter --output_dir results/benign_report

# Notes:
# - Uses config.py for ACTIVE_GRAPH_DIR and METADATA_PATH.
# - Only benign samples are analyzed.
# - Works directly on .pkl HeteroData graphs.
# """

# from __future__ import annotations

# import argparse
# import csv
# import math
# import os
# import pickle
# from collections import Counter, defaultdict
# from pathlib import Path
# from statistics import mean, median

# import config
# from graph_construction.pytorch_dataset import RansomwareGraphDataset

# # ---------- ReportLab ----------
# from reportlab.lib import colors
# from reportlab.lib.enums import TA_CENTER, TA_LEFT
# from reportlab.lib.pagesizes import A4
# from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
# from reportlab.lib.units import cm
# from reportlab.platypus import (
#     PageBreak,
#     Paragraph,
#     SimpleDocTemplate,
#     Spacer,
#     Table,
#     TableStyle,
# )

# # ============================================================
# #  CONFIG / LABEL LOGIC
# # ============================================================

# CLASS_BENIGN = 1

# BENIGN_OP_DESCRIPTIONS = {
#     "web_or_cloud_app": "Browser / cloud sync / web client",
#     "network_service": "Network daemon / background service",
#     "system_tool_or_installer": "System tool, updater, or installer",
#     "file_operation_app": "Document editor / file manager / archiver",
#     "system_configuration": "Registry / system config tool",
#     "gui_application": "GUI desktop application",
#     "crypto_or_security_tool": "Crypto library / security / AV helper",
#     "system_monitor_or_utility": "System monitor / diagnostic / CLI utility",
#     "simple_utility": "Simple utility (no strong signals)",
#     "unknown_parse_error": "Could not determine (graph parse error)",
# }

# API_PATTERNS = {
#     "network_communication": [
#         "internetopen", "httpsend", "httpopen", "wsaconnect", "wsasend",
#         "connect", "send", "recv", "getaddrinfo", "dnsquery",
#         "internetreadfile", "internetwritefile", "wsastartup",
#     ],
#     "file_io": [
#         "createfile", "readfile", "writefile", "deletefile", "movefile",
#         "copyfile", "openfile", "setfileattributes", "getfileattributes",
#         "findfile", "createdir", "removedir", "setendoffile",
#     ],
#     "registry_access": [
#         "regopen", "regcreate", "regset", "regquery", "regdelete",
#         "regclose", "regenumkey", "regenumvalue",
#     ],
#     "process_management": [
#         "createprocess", "openprocess", "terminateprocess", "shellexecute",
#         "createthread", "openthread", "suspendthread", "resumethread",
#         "virtualalloc", "virtualfree", "loadlibrary", "freelibrary",
#     ],
#     "crypto_or_encoding": [
#         "cryptgenrandom", "bcryptgenrandom", "cryptencrypt", "cryptdecrypt",
#         "crypthashdata", "base64", "rtldecompressbuffer",
#     ],
#     "system_info": [
#         "getsysteminfo", "getcomputername", "getusername", "getversion",
#         "gettickcount", "querysysteminfo", "ntquerysystem",
#     ],
#     "ui_or_display": [
#         "findwindow", "sendmessage", "postmessage", "setwindowtext",
#         "createwindow", "showwindow", "messagebox",
#     ],
# }

# BENIGN_TYPE_RULES = [
#     (lambda n, e, b: b.get("network_communication", 0) > 5 and b.get("file_io", 0) > 5,
#      "web_or_cloud_app"),
#     (lambda n, e, b: b.get("network_communication", 0) > 5,
#      "network_service"),
#     (lambda n, e, b: b.get("registry_access", 0) > 5 and b.get("process_management", 0) > 3,
#      "system_tool_or_installer"),
#     (lambda n, e, b: b.get("file_io", 0) > 5,
#      "file_operation_app"),
#     (lambda n, e, b: b.get("registry_access", 0) > 5,
#      "system_configuration"),
#     (lambda n, e, b: b.get("ui_or_display", 0) > 3,
#      "gui_application"),
#     (lambda n, e, b: b.get("crypto_or_encoding", 0) > 2,
#      "crypto_or_security_tool"),
#     (lambda n, e, b: b.get("system_info", 0) > 3,
#      "system_monitor_or_utility"),
# ]


# def classify_api_bucket(api_names: list[str]) -> Counter:
#     bucket_counts = Counter()
#     for name in api_names:
#         name_lower = str(name).lower()
#         for bucket, keywords in API_PATTERNS.items():
#             if any(kw in name_lower for kw in keywords):
#                 bucket_counts[bucket] += 1
#                 break
#     return bucket_counts


# def identify_benign_type(graph) -> dict:
#     try:
#         node_summary = {}
#         for nt in graph.node_types:
#             store = graph[nt]
#             if hasattr(store, "x") and store.x is not None:
#                 node_summary[nt] = int(store.x.shape[0])

#         edge_summary = {}
#         for et in graph.edge_types:
#             store = graph[et]
#             if hasattr(store, "edge_index") and store.edge_index is not None:
#                 edge_summary[str(et)] = int(store.edge_index.shape[1])

#         api_names = []
#         if "api" in graph.node_types and hasattr(graph["api"], "api_names"):
#             api_names = graph["api"].api_names or []

#         bucket_hits = classify_api_bucket(api_names)

#         nn = node_summary
#         has_network = nn.get("network", 0) > 2
#         has_file = nn.get("file", 0) > 5
#         has_reg = nn.get("registry", 0) > 3

#         op_type = "simple_utility"
#         for cond_fn, label in BENIGN_TYPE_RULES:
#             if cond_fn(nn, edge_summary, bucket_hits):
#                 op_type = label
#                 break
#         else:
#             if has_network and has_file:
#                 op_type = "web_or_cloud_app"
#             elif has_network:
#                 op_type = "network_service"
#             elif has_file and has_reg:
#                 op_type = "system_tool_or_installer"
#             elif has_file:
#                 op_type = "file_operation_app"
#             elif has_reg:
#                 op_type = "system_configuration"
#             else:
#                 op_type = "simple_utility"

#         top_apis = Counter([str(x).lower() for x in api_names]).most_common(10)

#         return {
#             "benign_op_type": op_type,
#             "node_summary": node_summary,
#             "edge_summary": edge_summary,
#             "api_bucket_hits": dict(bucket_hits),
#             "num_api_nodes": node_summary.get("api", 0),
#             "api_names_sample": list(api_names[:10]) if api_names else [],
#             "top_apis": top_apis,
#         }

#     except Exception as exc:
#         return {
#             "benign_op_type": "unknown_parse_error",
#             "error": str(exc),
#             "node_summary": {},
#             "edge_summary": {},
#             "api_bucket_hits": {},
#             "num_api_nodes": 0,
#             "api_names_sample": [],
#             "top_apis": [],
#         }


# # ============================================================
# #  ANALYSIS
# # ============================================================

# def safe_mean(values):
#     return round(mean(values), 2) if values else 0.0


# def safe_median(values):
#     return round(median(values), 2) if values else 0.0


# def safe_pct(count, total):
#     return round((count / total) * 100, 2) if total else 0.0


# def analyze_split(split: str) -> list[dict]:
#     dataset = RansomwareGraphDataset(
#         data_dir=str(config.DATA_DIR),
#         split=split,
#         cache_dir=str(config.ACTIVE_GRAPH_DIR),
#         validate_graphs=False,
#     )

#     print(f"[INFO] Scanning split: {dataset.split_dir} ({len(dataset.graph_files):,} graph files)")
#     rows = []

#     for i, graph_file in enumerate(dataset.graph_files, 1):
#         md5 = graph_file.stem
#         label = dataset.labels.get(md5, 2)
#         if label != CLASS_BENIGN:
#             continue

#         try:
#             with open(graph_file, "rb") as f:
#                 graph = pickle.load(f)
#         except Exception as exc:
#             rows.append({
#                 "split": split,
#                 "md5": md5,
#                 "benign_op_type": "unknown_parse_error",
#                 "description": BENIGN_OP_DESCRIPTIONS["unknown_parse_error"],
#                 "api_nodes": 0,
#                 "file_nodes": 0,
#                 "registry_nodes": 0,
#                 "process_nodes": 0,
#                 "network_nodes": 0,
#                 "temporal_edges": 0,
#                 "top_bucket": "",
#                 "top_bucket_count": 0,
#                 "top_api_1": "",
#                 "top_api_1_count": 0,
#                 "top_api_2": "",
#                 "top_api_2_count": 0,
#                 "top_api_3": "",
#                 "top_api_3_count": 0,
#                 "error": str(exc),
#             })
#             continue

#         info = identify_benign_type(graph)
#         node_summary = info["node_summary"]
#         bucket_hits = Counter(info["api_bucket_hits"])
#         top_bucket = bucket_hits.most_common(1)[0] if bucket_hits else ("", 0)
#         top_apis = info["top_apis"]

#         temporal_edges = 0
#         temporal_type = ("api", "temporal", "api")
#         if temporal_type in graph.edge_types and hasattr(graph[temporal_type], "edge_index"):
#             temporal_edges = int(graph[temporal_type].edge_index.shape[1])

#         def top_api_pair(idx):
#             if idx < len(top_apis):
#                 return top_apis[idx][0], top_apis[idx][1]
#             return "", 0

#         a1, a1c = top_api_pair(0)
#         a2, a2c = top_api_pair(1)
#         a3, a3c = top_api_pair(2)

#         rows.append({
#             "split": split,
#             "md5": md5,
#             "benign_op_type": info["benign_op_type"],
#             "description": BENIGN_OP_DESCRIPTIONS.get(info["benign_op_type"], info["benign_op_type"]),
#             "api_nodes": int(node_summary.get("api", 0)),
#             "file_nodes": int(node_summary.get("file", 0)),
#             "registry_nodes": int(node_summary.get("registry", 0)),
#             "process_nodes": int(node_summary.get("process", 0)),
#             "network_nodes": int(node_summary.get("network", 0)),
#             "temporal_edges": temporal_edges,
#             "top_bucket": top_bucket[0],
#             "top_bucket_count": int(top_bucket[1]),
#             "top_api_1": a1,
#             "top_api_1_count": int(a1c),
#             "top_api_2": a2,
#             "top_api_2_count": int(a2c),
#             "top_api_3": a3,
#             "top_api_3_count": int(a3c),
#             "error": "",
#         })

#         if i % 250 == 0:
#             print(f"  ... processed {i}/{len(dataset.graph_files)}")

#     return rows


# # ============================================================
# #  AGGREGATION
# # ============================================================

# def build_operation_summary(rows: list[dict]) -> list[dict]:
#     grouped = defaultdict(list)
#     for row in rows:
#         grouped[row["benign_op_type"]].append(row)

#     out = []
#     total = len(rows)

#     for op_type, items in sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True):
#         api_vals = [x["api_nodes"] for x in items]
#         file_vals = [x["file_nodes"] for x in items]
#         reg_vals = [x["registry_nodes"] for x in items]
#         proc_vals = [x["process_nodes"] for x in items]
#         net_vals = [x["network_nodes"] for x in items]
#         temporal_vals = [x["temporal_edges"] for x in items]

#         bucket_counter = Counter()
#         api_counter = Counter()
#         for x in items:
#             if x["top_bucket"]:
#                 bucket_counter[x["top_bucket"]] += 1
#             for api_key, api_count_key in [
#                 ("top_api_1", "top_api_1_count"),
#                 ("top_api_2", "top_api_2_count"),
#                 ("top_api_3", "top_api_3_count"),
#             ]:
#                 api_name = x[api_key]
#                 api_count = x[api_count_key]
#                 if api_name:
#                     api_counter[api_name] += api_count

#         out.append({
#             "benign_op_type": op_type,
#             "description": BENIGN_OP_DESCRIPTIONS.get(op_type, op_type),
#             "count": len(items),
#             "percentage": safe_pct(len(items), total),
#             "avg_api_nodes": safe_mean(api_vals),
#             "median_api_nodes": safe_median(api_vals),
#             "avg_file_nodes": safe_mean(file_vals),
#             "avg_registry_nodes": safe_mean(reg_vals),
#             "avg_process_nodes": safe_mean(proc_vals),
#             "avg_network_nodes": safe_mean(net_vals),
#             "avg_temporal_edges": safe_mean(temporal_vals),
#             "common_bucket_1": bucket_counter.most_common(1)[0][0] if bucket_counter else "",
#             "common_bucket_2": bucket_counter.most_common(2)[1][0] if len(bucket_counter) > 1 else "",
#             "common_api_1": api_counter.most_common(1)[0][0] if api_counter else "",
#             "common_api_2": api_counter.most_common(2)[1][0] if len(api_counter) > 1 else "",
#             "common_api_3": api_counter.most_common(3)[2][0] if len(api_counter) > 2 else "",
#         })
#     return out


# def build_split_summary(rows: list[dict]) -> list[dict]:
#     split_type_counter = defaultdict(Counter)
#     split_totals = Counter()

#     for row in rows:
#         split_type_counter[row["split"]][row["benign_op_type"]] += 1
#         split_totals[row["split"]] += 1

#     out = []
#     for split in ["train", "val", "test"]:
#         for op_type, count in split_type_counter[split].most_common():
#             out.append({
#                 "split": split,
#                 "benign_op_type": op_type,
#                 "description": BENIGN_OP_DESCRIPTIONS.get(op_type, op_type),
#                 "count": count,
#                 "percentage": safe_pct(count, split_totals[split]),
#             })
#     return out


# def build_representative_samples(rows: list[dict]) -> list[dict]:
#     grouped = defaultdict(list)
#     for row in rows:
#         grouped[row["benign_op_type"]].append(row)

#     out = []
#     for op_type, items in grouped.items():
#         api_vals = sorted([x["api_nodes"] for x in items])
#         med = median(api_vals) if api_vals else 0
#         ranked = sorted(items, key=lambda x: abs(x["api_nodes"] - med))
#         for sample in ranked[:5]:
#             out.append({
#                 "benign_op_type": op_type,
#                 "description": BENIGN_OP_DESCRIPTIONS.get(op_type, op_type),
#                 "split": sample["split"],
#                 "md5": sample["md5"],
#                 "api_nodes": sample["api_nodes"],
#                 "file_nodes": sample["file_nodes"],
#                 "registry_nodes": sample["registry_nodes"],
#                 "process_nodes": sample["process_nodes"],
#                 "network_nodes": sample["network_nodes"],
#                 "top_bucket": sample["top_bucket"],
#                 "top_api_1": sample["top_api_1"],
#                 "top_api_2": sample["top_api_2"],
#                 "top_api_3": sample["top_api_3"],
#             })
#     return sorted(out, key=lambda x: (x["benign_op_type"], x["split"], x["md5"]))


# # ============================================================
# #  CSV HELPERS
# # ============================================================

# def write_csv(path: Path, rows: list[dict]):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     if not rows:
#         with open(path, "w", newline="", encoding="utf-8") as f:
#             f.write("")
#         return
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#         writer.writeheader()
#         writer.writerows(rows)
#     print(f"[INFO] Saved: {path}")


# # ============================================================
# #  PDF REPORT
# # ============================================================

# def make_styles():
#     styles = getSampleStyleSheet()
#     styles.add(ParagraphStyle(
#         name="TitleX",
#         parent=styles["Title"],
#         fontName="Helvetica-Bold",
#         fontSize=20,
#         leading=24,
#         textColor=colors.HexColor("#17324D"),
#         alignment=TA_LEFT,
#         spaceAfter=10,
#     ))
#     styles.add(ParagraphStyle(
#         name="SubTitleX",
#         parent=styles["Normal"],
#         fontName="Helvetica",
#         fontSize=10,
#         leading=13,
#         textColor=colors.HexColor("#4A5D73"),
#         spaceAfter=10,
#     ))
#     styles.add(ParagraphStyle(
#         name="SectionX",
#         parent=styles["Heading2"],
#         fontName="Helvetica-Bold",
#         fontSize=13,
#         leading=16,
#         textColor=colors.HexColor("#17324D"),
#         spaceBefore=10,
#         spaceAfter=6,
#     ))
#     styles.add(ParagraphStyle(
#         name="BodyX",
#         parent=styles["BodyText"],
#         fontName="Helvetica",
#         fontSize=9.4,
#         leading=13,
#         alignment=TA_LEFT,
#         spaceAfter=5,
#     ))
#     styles.add(ParagraphStyle(
#         name="SmallX",
#         parent=styles["BodyText"],
#         fontName="Helvetica",
#         fontSize=8.2,
#         leading=10.5,
#         textColor=colors.HexColor("#374151"),
#     ))
#     return styles


# def draw_header_footer(canvas, doc):
#     canvas.saveState()
#     w, h = A4
#     canvas.setStrokeColor(colors.HexColor("#D6DEE7"))
#     canvas.line(1.6*cm, h-1.35*cm, w-1.6*cm, h-1.35*cm)
#     canvas.line(1.6*cm, 1.15*cm, w-1.6*cm, 1.15*cm)
#     canvas.setFont("Helvetica", 8)
#     canvas.setFillColor(colors.HexColor("#667085"))
#     canvas.drawString(1.6*cm, 0.72*cm, "Benign operation analysis report")
#     canvas.drawRightString(w-1.6*cm, 0.72*cm, f"Page {doc.page}")
#     canvas.restoreState()


# def make_table(data, col_widths=None, header_bg="#EAF1F8", font_size=8.5):
#     tbl = Table(data, colWidths=col_widths, repeatRows=1)
#     tbl.setStyle(TableStyle([
#         ("BACKGROUND", (0,0), (-1,0), colors.HexColor(header_bg)),
#         ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#17324D")),
#         ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
#         ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
#         ("FONTSIZE", (0,0), (-1,-1), font_size),
#         ("LEADING", (0,0), (-1,-1), font_size + 2),
#         ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#D8E1EA")),
#         ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
#         ("ALIGN", (0,0), (-1,0), "CENTER"),
#         ("LEFTPADDING", (0,0), (-1,-1), 6),
#         ("RIGHTPADDING", (0,0), (-1,-1), 6),
#         ("TOPPADDING", (0,0), (-1,-1), 5),
#         ("BOTTOMPADDING", (0,0), (-1,-1), 5),
#         ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
#     ]))
#     return tbl


# def generate_pdf_report(
#     pdf_path: Path,
#     rows: list[dict],
#     op_summary: list[dict],
#     split_summary: list[dict],
#     reps: list[dict],
#     selected_split: str,
# ):
#     styles = make_styles()

#     doc = SimpleDocTemplate(
#         str(pdf_path),
#         pagesize=A4,
#         leftMargin=1.7*cm,
#         rightMargin=1.7*cm,
#         topMargin=1.8*cm,
#         bottomMargin=1.5*cm,
#         title="Benign Operation Analysis Report",
#         author="OpenAI",
#     )
#     story = []

#     total = len(rows)
#     split_counts = Counter([r["split"] for r in rows])
#     type_counts = Counter([r["benign_op_type"] for r in rows])

#     story.append(Paragraph("Benign Operation Analysis Report", styles["TitleX"]))
#     story.append(Paragraph(
#         f"Graph source: {config.ACTIVE_GRAPH_SOURCE}<br/>"
#         f"Graph root: {config.ACTIVE_GRAPH_DIR}<br/>"
#         f"Metadata path: {config.METADATA_PATH}<br/>"
#         f"Scope: benign samples only | Split selection: {selected_split}",
#         styles["SubTitleX"]
#     ))
#     story.append(Spacer(1, 0.2*cm))

#     story.append(Paragraph("Executive summary", styles["SectionX"]))
#     summary_text = (
#         f"A total of <b>{total}</b> benign samples were analyzed. "
#         f"The most frequent benign operation class is <b>{type_counts.most_common(1)[0][0]}</b> "
#         f"with <b>{type_counts.most_common(1)[0][1]}</b> samples. "
#         f"Split coverage is train={split_counts.get('train', 0)}, "
#         f"val={split_counts.get('val', 0)}, test={split_counts.get('test', 0)}."
#     )
#     story.append(Paragraph(summary_text, styles["BodyX"]))

#     exec_tbl = [
#         ["Metric", "Value"],
#         ["Total benign samples", f"{total:,}"],
#         ["Train benign samples", f"{split_counts.get('train', 0):,}"],
#         ["Val benign samples", f"{split_counts.get('val', 0):,}"],
#         ["Test benign samples", f"{split_counts.get('test', 0):,}"],
#         ["Distinct benign operation types", str(len(type_counts))],
#         ["Most common type", f"{type_counts.most_common(1)[0][0]} ({type_counts.most_common(1)[0][1]})"],
#     ]
#     story.append(make_table(exec_tbl, col_widths=[6.5*cm, 9.3*cm]))
#     story.append(Spacer(1, 0.2*cm))

#     story.append(Paragraph("Overall operation distribution", styles["SectionX"]))
#     tdata = [["Operation type", "Description", "Count", "%", "Avg API", "Median API"]]
#     for row in op_summary:
#         tdata.append([
#             row["benign_op_type"],
#             row["description"],
#             str(row["count"]),
#             f'{row["percentage"]:.2f}',
#             f'{row["avg_api_nodes"]:.2f}',
#             f'{row["median_api_nodes"]:.2f}',
#         ])
#     story.append(make_table(tdata, col_widths=[4.0*cm, 6.3*cm, 1.5*cm, 1.2*cm, 1.6*cm, 1.7*cm], font_size=8.1))
#     story.append(Spacer(1, 0.2*cm))

#     story.append(Paragraph("Split-wise distribution", styles["SectionX"]))
#     sdata = [["Split", "Operation type", "Count", "%"]]
#     for row in split_summary:
#         sdata.append([
#             row["split"],
#             row["benign_op_type"],
#             str(row["count"]),
#             f'{row["percentage"]:.2f}',
#         ])
#     story.append(make_table(sdata, col_widths=[2.2*cm, 8.8*cm, 2.2*cm, 2.0*cm], font_size=8.2))
#     story.append(PageBreak())

#     story.append(Paragraph("Operation profiles", styles["SectionX"]))
#     story.append(Paragraph(
#         "The table below summarizes the structural profile of each benign operation type using average node and temporal-edge counts, along with the most characteristic API bucket and common API calls.",
#         styles["BodyX"]
#     ))
#     pdata = [[
#         "Type", "Avg API", "Avg File", "Avg Reg", "Avg Proc", "Avg Net",
#         "Avg Temp", "Common bucket", "Common APIs"
#     ]]
#     for row in op_summary:
#         common_apis = ", ".join([x for x in [row["common_api_1"], row["common_api_2"], row["common_api_3"]] if x])
#         pdata.append([
#             row["benign_op_type"],
#             f'{row["avg_api_nodes"]:.1f}',
#             f'{row["avg_file_nodes"]:.1f}',
#             f'{row["avg_registry_nodes"]:.1f}',
#             f'{row["avg_process_nodes"]:.1f}',
#             f'{row["avg_network_nodes"]:.1f}',
#             f'{row["avg_temporal_edges"]:.1f}',
#             row["common_bucket_1"],
#             common_apis,
#         ])
#     story.append(make_table(
#         pdata,
#         col_widths=[3.3*cm, 1.35*cm, 1.35*cm, 1.35*cm, 1.35*cm, 1.35*cm, 1.45*cm, 2.6*cm, 3.0*cm],
#         font_size=7.4
#     ))
#     story.append(Spacer(1, 0.2*cm))

#     story.append(Paragraph("Representative benign samples", styles["SectionX"]))
#     story.append(Paragraph(
#         "For each benign operation type, five representative samples closest to the median API-node size are listed below.",
#         styles["BodyX"]
#     ))
#     rdata = [["Type", "Split", "MD5", "API", "File", "Reg", "Proc", "Net", "Top bucket", "Top API"]]
#     for row in reps[:60]:
#         rdata.append([
#             row["benign_op_type"],
#             row["split"],
#             row["md5"],
#             str(row["api_nodes"]),
#             str(row["file_nodes"]),
#             str(row["registry_nodes"]),
#             str(row["process_nodes"]),
#             str(row["network_nodes"]),
#             row["top_bucket"],
#             row["top_api_1"],
#         ])
#     story.append(make_table(
#         rdata,
#         col_widths=[3.0*cm, 1.2*cm, 3.2*cm, 0.9*cm, 0.9*cm, 0.9*cm, 0.9*cm, 0.9*cm, 2.3*cm, 2.4*cm],
#         font_size=7.2
#     ))
#     story.append(PageBreak())

#     story.append(Paragraph("Interpretation notes", styles["SectionX"]))
#     notes = [
#         "The report classifies only benign samples; ransomware and other-malware samples are ignored.",
#         "Operation types are heuristic labels inferred from graph structure and API-call patterns.",
#         "API-run compression or behavior-aware capping in the training pipeline does not affect this report unless the analyzed .pkl graphs themselves were permanently modified beforehand.",
#         "The most reliable use of this report is to document the benign behavioral coverage seen by the model during training and evaluation."
#     ]
#     for note in notes:
#         story.append(Paragraph(f"• {note}", styles["BodyX"]))

#     doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
#     print(f"[INFO] Saved: {pdf_path}")


# # ============================================================
# #  MAIN
# # ============================================================

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
#     p.add_argument("--output_dir", default="benign_operation_report")
#     return p.parse_args()


# def main():
#     args = parse_args()
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     splits = ["train", "val", "test"] if args.split == "all" else [args.split]

#     all_rows = []
#     for split in splits:
#         split_rows = analyze_split(split)
#         all_rows.extend(split_rows)

#     op_summary = build_operation_summary(all_rows)
#     split_summary = build_split_summary(all_rows)
#     reps = build_representative_samples(all_rows)

#     write_csv(output_dir / "benign_per_sample.csv", all_rows)
#     write_csv(output_dir / "benign_operation_summary.csv", op_summary)
#     write_csv(output_dir / "benign_split_summary.csv", split_summary)
#     write_csv(output_dir / "benign_representative_samples.csv", reps)

#     pdf_name = f"benign_operation_report_{args.split}.pdf"
#     generate_pdf_report(
#         pdf_path=output_dir / pdf_name,
#         rows=all_rows,
#         op_summary=op_summary,
#         split_summary=split_summary,
#         reps=reps,
#         selected_split=args.split,
#     )

#     print("\n" + "=" * 70)
#     print(f"BENIGN REPORT COMPLETE | scope={args.split} | total benign={len(all_rows)}")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
scripts/benign_operation_reporter_improved.py

Scans the enriched TGNN graph cache, analyzes ONLY benign samples, and produces:
1) benign_per_sample.csv               -> rich per-sample behavioral report
2) benign_operation_summary.csv        -> aggregated operation-type summary
3) benign_split_summary.csv            -> per-split distribution summary
4) benign_representative_samples.csv   -> representative samples per benign type
5) benign_per_sample.jsonl             -> full structured per-sample records
6) benign_operation_report_<split>.pdf -> detailed PDF report

Usage:
    python -m scripts.benign_operation_reporter_improved
    python -m scripts.benign_operation_reporter_improved --split train
    python -m scripts.benign_operation_reporter_improved --split all --output_dir results/benign_report

Notes:
- Uses config.py for ACTIVE_GRAPH_DIR and METADATA_PATH.
- Only benign samples are analyzed.
- Works directly on .pkl HeteroData graphs.
- If --split all is used, it takes benign samples from train, val, and test.
- This is still heuristic profiling, but much richer than a single coarse benign label.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple

import config
from graph_construction.pytorch_dataset import RansomwareGraphDataset

# ---------- ReportLab ----------
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
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
# CONFIG / LABEL LOGIC
# ============================================================

CLASS_BENIGN = 1

BENIGN_OP_DESCRIPTIONS = {
    "web_or_cloud_app": "Browser, sync client, or web-connected desktop app",
    "network_service": "Background network service or daemon",
    "system_tool_or_installer": "Installer, updater, or system-management utility",
    "file_operation_app": "Document editor, file manager, archiver, or file-heavy tool",
    "system_configuration": "Registry or system configuration utility",
    "gui_application": "Interactive GUI-focused desktop application",
    "crypto_or_security_tool": "Security, antivirus, cryptographic, or encoding helper",
    "system_monitor_or_utility": "Monitoring, diagnostics, or information utility",
    "simple_utility": "Simple utility with weak or mixed signals",
    "unknown_parse_error": "Could not determine due to graph loading/parsing issue",
}

API_PATTERNS = {
    "network_communication": [
        "internetopen", "httpopen", "httpsend", "internetreadfile", "internetwritefile",
        "wsastartup", "connect", "wsaconnect", "send", "wsasend", "recv",
        "getaddrinfo", "dnsquery", "socket",
    ],
    "file_io": [
        "createfile", "openfile", "readfile", "writefile", "deletefile",
        "copyfile", "movefile", "findfile", "setfileattributes", "getfileattributes",
        "createdirectory", "removedirectory", "createdir", "removedir", "setendoffile",
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

ACTION_FAMILIES = {
    "file_read_write": ["createfile", "openfile", "readfile", "writefile", "copyfile", "movefile"],
    "file_delete_or_attr": ["deletefile", "setfileattributes", "getfileattributes", "setendoffile"],
    "registry_read_write": ["regopen", "regquery", "regset", "regcreate", "regdelete"],
    "process_spawn_control": ["createprocess", "openprocess", "terminateprocess", "shellexecute"],
    "thread_control": ["createthread", "openthread", "suspendthread", "resumethread"],
    "memory_module": ["virtualalloc", "virtualfree", "loadlibrary", "freelibrary", "getprocaddress"],
    "network_session": ["internetopen", "httpopen", "httpsend", "internetreadfile", "internetwritefile"],
    "socket_comm": ["socket", "connect", "send", "recv", "wsastartup", "wsaconnect", "wsasend"],
    "crypto_random_hash": ["cryptgenrandom", "bcryptgenrandom", "cryptencrypt", "cryptdecrypt", "crypthashdata"],
    "system_discovery": ["getsysteminfo", "getcomputername", "getusername", "getversion", "gettickcount"],
    "ui_interaction": ["findwindow", "sendmessage", "postmessage", "setwindowtext", "showwindow", "messagebox"],
}

BENIGN_TYPE_RULES = [
    (lambda n, e, b, a: b.get("network_communication", 0) > 5 and b.get("file_io", 0) > 5,
     "web_or_cloud_app", "Strong network plus file activity suggests browser, sync, or web-connected app behavior."),
    (lambda n, e, b, a: b.get("network_communication", 0) > 5,
     "network_service", "Heavy network activity with weaker non-network signals suggests service-like benign behavior."),
    (lambda n, e, b, a: b.get("registry_access", 0) > 5 and b.get("process_management", 0) > 3,
     "system_tool_or_installer", "Registry changes plus process-management activity suggest installer or updater behavior."),
    (lambda n, e, b, a: b.get("file_io", 0) > 5,
     "file_operation_app", "File operations dominate this sample, indicating document or file-handling behavior."),
    (lambda n, e, b, a: b.get("registry_access", 0) > 5,
     "system_configuration", "Registry access dominates, consistent with system configuration tools."),
    (lambda n, e, b, a: b.get("ui_or_display", 0) > 3,
     "gui_application", "UI-related calls are prominent, consistent with interactive desktop applications."),
    (lambda n, e, b, a: b.get("crypto_or_encoding", 0) > 2,
     "crypto_or_security_tool", "Cryptographic or encoding APIs are unusually prominent for this sample."),
    (lambda n, e, b, a: b.get("system_info", 0) > 3,
     "system_monitor_or_utility", "System-information calls dominate, suggesting monitoring or diagnostic behavior."),
]

EDGE_LABELS = {
    "('api', 'temporal', 'api')": "api_temporal_api",
    "('api', 'operates_on', 'file')": "api_file",
    "('api', 'modifies', 'registry')": "api_registry",
    "('process', 'executes', 'api')": "process_api",
    "('api', 'connects_to', 'network')": "api_network",
}


# ============================================================
# HELPERS
# ============================================================

def safe_mean(values: List[float]) -> float:
    return round(mean(values), 2) if values else 0.0


def safe_median(values: List[float]) -> float:
    return round(median(values), 2) if values else 0.0


def safe_pct(count: int, total: int) -> float:
    return round((count / total) * 100.0, 2) if total else 0.0


def normalize_api_name(x) -> str:
    return str(x).strip().lower() if x is not None else ""


def counter_to_percentages(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: round((v / total) * 100.0, 2) for k, v in counter.items()}


def top_k_pairs(counter_like, k: int = 3) -> List[Tuple[str, int]]:
    if not counter_like:
        return []
    if isinstance(counter_like, Counter):
        return counter_like.most_common(k)
    return Counter(counter_like).most_common(k)


# ============================================================
# BEHAVIOR ANALYSIS
# ============================================================

def classify_api_bucket(api_names: List[str]) -> Counter:
    bucket_counts = Counter()
    for name in api_names:
        name_lower = normalize_api_name(name)
        for bucket, keywords in API_PATTERNS.items():
            if any(kw in name_lower for kw in keywords):
                bucket_counts[bucket] += 1
                break
    return bucket_counts


def classify_action_families(api_names: List[str]) -> Counter:
    action_counts = Counter()
    for name in api_names:
        name_lower = normalize_api_name(name)
        for family, keywords in ACTION_FAMILIES.items():
            if any(kw in name_lower for kw in keywords):
                action_counts[family] += 1
                break
    return action_counts


def collect_edge_summary(graph) -> Dict[str, int]:
    edge_summary = {}
    for et in graph.edge_types:
        store = graph[et]
        if hasattr(store, "edge_index") and store.edge_index is not None:
            key = str(et)
            edge_summary[key] = int(store.edge_index.shape[1])
    return edge_summary


def collect_node_summary(graph) -> Dict[str, int]:
    node_summary = {}
    for nt in graph.node_types:
        store = graph[nt]
        if hasattr(store, "x") and store.x is not None:
            node_summary[nt] = int(store.x.shape[0])
    return node_summary


def infer_behavior_tags(node_summary: Dict[str, int], bucket_hits: Counter, action_hits: Counter, edge_summary: Dict[str, int]) -> List[str]:
    tags = []

    if bucket_hits.get("file_io", 0) > 0:
        tags.append("file_io")
    if bucket_hits.get("network_communication", 0) > 0:
        tags.append("network_activity")
    if bucket_hits.get("registry_access", 0) > 0:
        tags.append("registry_activity")
    if bucket_hits.get("process_management", 0) > 0:
        tags.append("process_management")
    if bucket_hits.get("system_info", 0) > 0:
        tags.append("system_info_collection")
    if bucket_hits.get("ui_or_display", 0) > 0:
        tags.append("ui_interaction")
    if bucket_hits.get("crypto_or_encoding", 0) > 0:
        tags.append("crypto_or_encoding")

    if node_summary.get("network", 0) > 0 and "network_activity" not in tags:
        tags.append("network_context_present")
    if node_summary.get("file", 0) > 0 and "file_io" not in tags:
        tags.append("file_context_present")
    if node_summary.get("registry", 0) > 0 and "registry_activity" not in tags:
        tags.append("registry_context_present")

    process_api_edges = edge_summary.get("('process', 'executes', 'api')", 0)
    if process_api_edges > 0:
        tags.append("process_exec_context")

    if action_hits.get("network_session", 0) + action_hits.get("socket_comm", 0) >= 3:
        tags.append("active_communications")
    if action_hits.get("file_read_write", 0) >= 3:
        tags.append("active_file_handling")
    if action_hits.get("registry_read_write", 0) >= 3:
        tags.append("active_registry_handling")

    return sorted(set(tags))


def summarize_behavior_profile(bucket_hits: Counter) -> str:
    if not bucket_hits:
        return "No strong API bucket signal detected."

    parts = []
    percentages = counter_to_percentages(bucket_hits)
    for bucket, pct in sorted(percentages.items(), key=lambda kv: kv[1], reverse=True)[:3]:
        parts.append(f"{bucket}={pct:.1f}%")
    return ", ".join(parts)


def infer_confidence(bucket_hits: Counter, node_summary: Dict[str, int], op_type: str) -> str:
    top_bucket_count = max(bucket_hits.values()) if bucket_hits else 0
    total_signal = sum(bucket_hits.values())

    if op_type == "unknown_parse_error":
        return "low"
    if total_signal >= 12 and top_bucket_count >= 5:
        return "high"
    if total_signal >= 5:
        return "medium"
    if node_summary.get("api", 0) >= 20:
        return "medium"
    return "low"


def generate_behavior_summary(op_type: str, behavior_tags: List[str], bucket_hits: Counter, action_hits: Counter, node_summary: Dict[str, int]) -> str:
    dominant_buckets = [x[0] for x in bucket_hits.most_common(2)]
    dominant_actions = [x[0] for x in action_hits.most_common(2)]

    fragments = []

    if dominant_buckets:
        fragments.append(f"Dominant activity is {', '.join(dominant_buckets)}")
    if dominant_actions:
        fragments.append(f"main action families are {', '.join(dominant_actions)}")
    if behavior_tags:
        fragments.append(f"behavior tags: {', '.join(behavior_tags[:5])}")

    nn = node_summary
    context_bits = []
    if nn.get("file", 0) > 0:
        context_bits.append(f"file_nodes={nn.get('file', 0)}")
    if nn.get("registry", 0) > 0:
        context_bits.append(f"registry_nodes={nn.get('registry', 0)}")
    if nn.get("process", 0) > 0:
        context_bits.append(f"process_nodes={nn.get('process', 0)}")
    if nn.get("network", 0) > 0:
        context_bits.append(f"network_nodes={nn.get('network', 0)}")

    context_text = f" Context includes {'; '.join(context_bits)}." if context_bits else ""
    prefix = BENIGN_OP_DESCRIPTIONS.get(op_type, op_type)

    if fragments:
        return f"Best-fit benign type: {prefix}. " + "; ".join(fragments) + "." + context_text
    return f"Best-fit benign type: {prefix}. Limited behavior evidence available.{context_text}"


def identify_benign_type(graph) -> dict:
    try:
        node_summary = collect_node_summary(graph)
        edge_summary = collect_edge_summary(graph)

        api_names = []
        if "api" in graph.node_types and hasattr(graph["api"], "api_names"):
            api_names = list(graph["api"].api_names or [])

        bucket_hits = classify_api_bucket(api_names)
        action_hits = classify_action_families(api_names)

        nn = node_summary
        has_network = nn.get("network", 0) > 2
        has_file = nn.get("file", 0) > 5
        has_reg = nn.get("registry", 0) > 3

        op_type = "simple_utility"
        reason = "Weak or mixed signals, defaulting to simple utility."
        for cond_fn, label, label_reason in BENIGN_TYPE_RULES:
            if cond_fn(nn, edge_summary, bucket_hits, action_hits):
                op_type = label
                reason = label_reason
                break
        else:
            if has_network and has_file:
                op_type = "web_or_cloud_app"
                reason = "Fallback rule: both file and network contexts are prominent."
            elif has_network:
                op_type = "network_service"
                reason = "Fallback rule: network context dominates."
            elif has_file and has_reg:
                op_type = "system_tool_or_installer"
                reason = "Fallback rule: file and registry contexts co-occur strongly."
            elif has_file:
                op_type = "file_operation_app"
                reason = "Fallback rule: file context dominates."
            elif has_reg:
                op_type = "system_configuration"
                reason = "Fallback rule: registry context dominates."

        top_apis = Counter([normalize_api_name(x) for x in api_names]).most_common(10)
        behavior_tags = infer_behavior_tags(nn, bucket_hits, action_hits, edge_summary)
        bucket_percentages = counter_to_percentages(bucket_hits)
        action_percentages = counter_to_percentages(action_hits)
        confidence = infer_confidence(bucket_hits, nn, op_type)
        behavior_summary = generate_behavior_summary(op_type, behavior_tags, bucket_hits, action_hits, nn)

        unique_api_count = len(set(normalize_api_name(x) for x in api_names if normalize_api_name(x)))
        total_api_count = len(api_names)
        repeated_api_ratio = round(1.0 - (unique_api_count / total_api_count), 4) if total_api_count > 0 else 0.0
        active_behavior_buckets = len([k for k, v in bucket_hits.items() if v > 0])

        edge_type_counts = {
            EDGE_LABELS.get(k, k): v for k, v in edge_summary.items()
        }

        return {
            "benign_op_type": op_type,
            "classification_reason": reason,
            "confidence": confidence,
            "behavior_summary": behavior_summary,
            "behavior_tags": behavior_tags,
            "node_summary": node_summary,
            "edge_summary": edge_summary,
            "edge_type_counts": edge_type_counts,
            "api_bucket_hits": dict(bucket_hits),
            "api_bucket_percentages": bucket_percentages,
            "action_family_hits": dict(action_hits),
            "action_family_percentages": action_percentages,
            "num_api_nodes": node_summary.get("api", 0),
            "unique_api_count": unique_api_count,
            "repeated_api_ratio": repeated_api_ratio,
            "active_behavior_buckets": active_behavior_buckets,
            "api_names_sample": list(api_names[:10]) if api_names else [],
            "top_apis": top_apis,
            "top_action_families": action_hits.most_common(5),
        }

    except Exception as exc:
        return {
            "benign_op_type": "unknown_parse_error",
            "classification_reason": f"Graph parse/analysis failed: {exc}",
            "confidence": "low",
            "behavior_summary": "Could not analyze this sample because graph parsing or inspection failed.",
            "behavior_tags": [],
            "error": str(exc),
            "node_summary": {},
            "edge_summary": {},
            "edge_type_counts": {},
            "api_bucket_hits": {},
            "api_bucket_percentages": {},
            "action_family_hits": {},
            "action_family_percentages": {},
            "num_api_nodes": 0,
            "unique_api_count": 0,
            "repeated_api_ratio": 0.0,
            "active_behavior_buckets": 0,
            "api_names_sample": [],
            "top_apis": [],
            "top_action_families": [],
        }


# ============================================================
# ANALYSIS
# ============================================================

def analyze_split(split: str) -> List[dict]:
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
                "behavior_summary": "Could not load graph file.",
                "classification_reason": str(exc),
                "confidence": "low",
                "behavior_tags": "",
                "api_nodes": 0,
                "file_nodes": 0,
                "registry_nodes": 0,
                "process_nodes": 0,
                "network_nodes": 0,
                "temporal_edges": 0,
                "process_api_edges": 0,
                "api_file_edges": 0,
                "api_registry_edges": 0,
                "api_network_edges": 0,
                "unique_api_count": 0,
                "repeated_api_ratio": 0.0,
                "active_behavior_buckets": 0,
                "top_bucket": "",
                "top_bucket_count": 0,
                "bucket_profile": "",
                "top_action_family_1": "",
                "top_action_family_2": "",
                "top_action_family_3": "",
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
        top_actions = info["top_action_families"]
        edge_counts = info.get("edge_type_counts", {})

        def top_api_pair(idx: int):
            if idx < len(top_apis):
                return top_apis[idx][0], top_apis[idx][1]
            return "", 0

        def top_action(idx: int):
            if idx < len(top_actions):
                return top_actions[idx][0]
            return ""

        a1, a1c = top_api_pair(0)
        a2, a2c = top_api_pair(1)
        a3, a3c = top_api_pair(2)

        rows.append({
            "split": split,
            "md5": md5,
            "benign_op_type": info["benign_op_type"],
            "description": BENIGN_OP_DESCRIPTIONS.get(info["benign_op_type"], info["benign_op_type"]),
            "behavior_summary": info["behavior_summary"],
            "classification_reason": info["classification_reason"],
            "confidence": info["confidence"],
            "behavior_tags": " | ".join(info["behavior_tags"]),
            "api_nodes": int(node_summary.get("api", 0)),
            "file_nodes": int(node_summary.get("file", 0)),
            "registry_nodes": int(node_summary.get("registry", 0)),
            "process_nodes": int(node_summary.get("process", 0)),
            "network_nodes": int(node_summary.get("network", 0)),
            "temporal_edges": int(edge_counts.get("api_temporal_api", 0)),
            "process_api_edges": int(edge_counts.get("process_api", 0)),
            "api_file_edges": int(edge_counts.get("api_file", 0)),
            "api_registry_edges": int(edge_counts.get("api_registry", 0)),
            "api_network_edges": int(edge_counts.get("api_network", 0)),
            "unique_api_count": int(info.get("unique_api_count", 0)),
            "repeated_api_ratio": float(info.get("repeated_api_ratio", 0.0)),
            "active_behavior_buckets": int(info.get("active_behavior_buckets", 0)),
            "top_bucket": top_bucket[0],
            "top_bucket_count": int(top_bucket[1]),
            "bucket_profile": summarize_behavior_profile(bucket_hits),
            "top_action_family_1": top_action(0),
            "top_action_family_2": top_action(1),
            "top_action_family_3": top_action(2),
            "top_api_1": a1,
            "top_api_1_count": int(a1c),
            "top_api_2": a2,
            "top_api_2_count": int(a2c),
            "top_api_3": a3,
            "top_api_3_count": int(a3c),
            "error": info.get("error", ""),
        })

        if i % 250 == 0:
            print(f"  ... processed {i}/{len(dataset.graph_files)}")

    return rows


# ============================================================
# AGGREGATION
# ============================================================

def build_operation_summary(rows: List[dict]) -> List[dict]:
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
        unique_api_vals = [x["unique_api_count"] for x in items]
        repeat_vals = [x["repeated_api_ratio"] for x in items]

        bucket_counter = Counter()
        action_counter = Counter()
        api_counter = Counter()
        conf_counter = Counter()

        for x in items:
            if x["top_bucket"]:
                bucket_counter[x["top_bucket"]] += 1
            for af in ["top_action_family_1", "top_action_family_2", "top_action_family_3"]:
                if x[af]:
                    action_counter[x[af]] += 1
            for api_key, api_count_key in [
                ("top_api_1", "top_api_1_count"),
                ("top_api_2", "top_api_2_count"),
                ("top_api_3", "top_api_3_count"),
            ]:
                api_name = x[api_key]
                api_count = x[api_count_key]
                if api_name:
                    api_counter[api_name] += api_count
            conf_counter[x.get("confidence", "unknown")] += 1

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
            "avg_unique_api_count": safe_mean(unique_api_vals),
            "avg_repeated_api_ratio": safe_mean(repeat_vals),
            "common_bucket_1": bucket_counter.most_common(1)[0][0] if bucket_counter else "",
            "common_bucket_2": bucket_counter.most_common(2)[1][0] if len(bucket_counter) > 1 else "",
            "common_action_1": action_counter.most_common(1)[0][0] if action_counter else "",
            "common_action_2": action_counter.most_common(2)[1][0] if len(action_counter) > 1 else "",
            "common_api_1": api_counter.most_common(1)[0][0] if api_counter else "",
            "common_api_2": api_counter.most_common(2)[1][0] if len(api_counter) > 1 else "",
            "common_api_3": api_counter.most_common(3)[2][0] if len(api_counter) > 2 else "",
            "confidence_mode": conf_counter.most_common(1)[0][0] if conf_counter else "",
        })
    return out


def build_split_summary(rows: List[dict]) -> List[dict]:
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


def build_representative_samples(rows: List[dict]) -> List[dict]:
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
                "top_action_family_1": sample["top_action_family_1"],
                "top_api_1": sample["top_api_1"],
                "behavior_summary": sample["behavior_summary"],
                "confidence": sample["confidence"],
            })
    return sorted(out, key=lambda x: (x["benign_op_type"], x["split"], x["md5"]))


# ============================================================
# OUTPUT HELPERS
# ============================================================

def write_csv(path: Path, rows: List[dict]):
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


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved: {path}")


# ============================================================
# PDF REPORT
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
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17324D")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8E1EA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    return tbl


def generate_pdf_report(pdf_path: Path, rows: List[dict], op_summary: List[dict], split_summary: List[dict], reps: List[dict], selected_split: str):
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
    top_type = type_counts.most_common(1)[0] if type_counts else ("none", 0)
    summary_text = (
        f"A total of <b>{total}</b> benign samples were analyzed. "
        f"The most frequent benign operation class is <b>{top_type[0]}</b> "
        f"with <b>{top_type[1]}</b> samples. "
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
        ["Most common type", f"{top_type[0]} ({top_type[1]})"],
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
            f"{row['percentage']:.2f}",
            f"{row['avg_api_nodes']:.2f}",
            f"{row['median_api_nodes']:.2f}",
        ])
    story.append(make_table(tdata, col_widths=[4.0*cm, 6.3*cm, 1.5*cm, 1.2*cm, 1.6*cm, 1.7*cm], font_size=8.1))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Split-wise distribution", styles["SectionX"]))
    sdata = [["Split", "Operation type", "Count", "%"]]
    for row in split_summary:
        sdata.append([row["split"], row["benign_op_type"], str(row["count"]), f"{row['percentage']:.2f}"])
    story.append(make_table(sdata, col_widths=[2.2*cm, 8.8*cm, 2.2*cm, 2.0*cm], font_size=8.2))
    story.append(PageBreak())

    story.append(Paragraph("Operation profiles", styles["SectionX"]))
    pdata = [[
        "Type", "Avg API", "Avg File", "Avg Reg", "Avg Proc", "Avg Net",
        "Avg Temp", "Common bucket", "Common action"
    ]]
    for row in op_summary:
        pdata.append([
            row["benign_op_type"],
            f"{row['avg_api_nodes']:.1f}",
            f"{row['avg_file_nodes']:.1f}",
            f"{row['avg_registry_nodes']:.1f}",
            f"{row['avg_process_nodes']:.1f}",
            f"{row['avg_network_nodes']:.1f}",
            f"{row['avg_temporal_edges']:.1f}",
            row["common_bucket_1"],
            row["common_action_1"],
        ])
    story.append(make_table(
        pdata,
        col_widths=[3.2*cm, 1.2*cm, 1.2*cm, 1.2*cm, 1.2*cm, 1.2*cm, 1.3*cm, 2.7*cm, 3.1*cm],
        font_size=7.4,
    ))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Representative benign samples", styles["SectionX"]))
    rdata = [["Type", "Split", "MD5", "API", "Top bucket", "Top action", "Top API", "Confidence"]]
    for row in reps[:60]:
        rdata.append([
            row["benign_op_type"],
            row["split"],
            row["md5"],
            str(row["api_nodes"]),
            row["top_bucket"],
            row["top_action_family_1"],
            row["top_api_1"],
            row["confidence"],
        ])
    story.append(make_table(
        rdata,
        col_widths=[3.0*cm, 1.2*cm, 3.3*cm, 1.0*cm, 2.2*cm, 2.6*cm, 2.6*cm, 1.7*cm],
        font_size=7.2,
    ))
    story.append(PageBreak())

    story.append(Paragraph("Interpretation notes", styles["SectionX"]))
    notes = [
        "The report classifies only benign samples; ransomware and other-malware samples are ignored.",
        "Operation types are heuristic labels inferred from graph structure and API-call patterns.",
        "Each sample now includes behavior tags, action-family summaries, edge-type counts, and a short narrative explanation.",
        "If --split all is used, the script takes all benign samples from train, val, and test.",
    ]
    for note in notes:
        story.append(Paragraph(f"• {note}", styles["BodyX"]))

    doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
    print(f"[INFO] Saved: {pdf_path}")


# ============================================================
# MAIN
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
    write_jsonl(output_dir / "benign_per_sample.jsonl", all_rows)

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

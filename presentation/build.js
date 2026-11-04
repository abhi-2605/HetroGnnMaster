const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.APPLY_THEME || "/Users/abhishek/.claude/skills/synced/51c3770a-812b-47d9-b253-59239ab6072d_2159f811-5731-4d11-887b-afc083741433/pptx/scripts/apply_theme.js");

const A = __dirname + "/assets/";
const OUT = __dirname + "/Thesis_Defence_Menon_editable.pptx";

// ---------- theme ----------
const HEX = {
  red: "C61A27", ink: "1F2328", grey: "5B6670", light: "F3F4F6", line: "D9DDE1",
  sem: "0072B2", str: "E69F00", tmp: "009E73", base: "C9CED4", attacked: "5B6670",
  redTint: "FBE9EA", semTint: "E5F1F8", strTint: "FDF3E0", tmpTint: "E0F3EE",
};
const THEME = {
  name: "Koblenz Thesis Defence", headFontFace: "Georgia", bodyFontFace: "Arial",
  colors: {
    dk1: HEX.ink, lt1: "FFFFFF", dk2: HEX.red, lt2: HEX.light,
    accent1: HEX.red, accent2: HEX.sem, accent3: HEX.str, accent4: HEX.tmp,
    accent5: HEX.grey, accent6: HEX.base, hlink: HEX.sem, folHlink: HEX.grey,
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Kandenghat Abhishek Menon";
pres.title = "Threat Detection Using Heterogeneous Graph Neural Networks with Semantic and Temporal Features";
const C = pres.SchemeColor;
// scheme aliases
const RED = C.accent1, SEM = C.accent2, STR = C.accent3, TMP = C.accent4, GREY = C.accent5, INK = C.text1, WHITE = C.background1, LIGHT = C.background2;

const W = 13.333, H = 7.5, MX = 0.6;

// ---------- icons ----------
async function icon(Comp, hex, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ---------- layouts ----------
const FOOT = "K. A. Menon  ·  Master's Thesis Defence  ·  Universität Koblenz";
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: HEX.red },
  objects: [],
});
pres.defineSlideMaster({
  title: "DIVIDER",
  background: { color: HEX.red },
  objects: [
    { image: { x: 10.75, y: 0.45, w: 2.0, h: 0.61, path: A + "logo_white.png" } },
    { placeholder: { options: { name: "num", type: "body", x: 1.0, y: 2.15, w: 4, h: 1.3, fontSize: 72, bold: true, color: WHITE, fontFace: "Georgia", margin: 0, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 1.0, y: 3.55, w: 10.5, h: 1.1, fontSize: 40, bold: true, color: WHITE, margin: 0, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "sub", type: "body", x: 1.0, y: 4.75, w: 10.5, h: 0.6, fontSize: 18, color: WHITE, margin: 0, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { image: { x: 11.33, y: 0.38, w: 1.4, h: 0.427, path: A + "logo.png" } },
    { placeholder: { options: { name: "kicker", type: "body", x: MX, y: 0.36, w: 10.3, h: 0.3, fontSize: 12, bold: true, color: GREY, charSpacing: 2, margin: 0, align: "left" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.66, w: 10.6, h: 0.72, fontSize: 28, bold: true, color: RED, margin: 0, valign: "middle", align: "left" }, text: "" } },
    { text: { text: FOOT, options: { x: MX, y: 7.0, w: 4.8, h: 0.28, fontSize: 10, color: GREY, margin: 0 } } },
  ],
  slideNumber: { x: 12.23, y: 7.0, w: 0.5, h: 0.28, fontSize: 10, color: GREY, align: "right", margin: 0 },
});
pres.defineSlideMaster({ title: "CLOSING", background: { color: HEX.red }, objects: [] });

// ---------- helpers ----------
let section = "";
function sec(t) { section = t; pres.addSection({ title: t }); }
const PARTS = ["Motivation", "Methodology", "Results", "Conclusion"];
function content(kicker, title, notes, part) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(kicker.toUpperCase(), { placeholder: "kicker" });
  // agenda tracker in the footer: the current part is highlighted
  s.addText(PARTS.map((p, i) => ({ text: (i ? "   ·   " : "") + String(i + 1).padStart(2, "0") + " " + p,
    options: { bold: i === part, color: i === part ? HEX.red : HEX.grey } })),
    { isTextBox: true, x: 5.4, y: 7.0, w: 6.6, h: 0.28, fontSize: 10, margin: 0, align: "right", objectName: "agenda tracker" });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function divider(num, title, sub, notes) {
  const s = pres.addSlide({ masterName: "DIVIDER", sectionTitle: section });
  s.addText(num, { placeholder: "num" });
  s.addText(title, { placeholder: "title" });
  if (sub) s.addText(sub, { placeholder: "sub" });
  if (notes) s.addNotes(notes);
  return s;
}
const shadow = () => ({ type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 });
function card(s, x, y, w, h, fill, name, withShadow = false) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill || LIGHT }, line: { color: fill || LIGHT, width: 0 }, rectRadius: 0.08, objectName: name || "card", shadow: withShadow ? shadow() : undefined });
}
function txt(s, text, o) { s.addText(text, Object.assign({ isTextBox: true, margin: 0, valign: "top", fontSize: 14, color: INK }, o)); }
function line(s, x1, y1, x2, y2, color, width = 1.5, arrow = false, name = "connector") {
  // arrows are only used on left-to-right connectors
  const o = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.001), h: Math.max(Math.abs(y2 - y1), 0.001), line: { color, width }, objectName: name };
  if ((x2 < x1) !== (y2 < y1)) o.flipH = true;
  if (arrow) o.line.endArrowType = "triangle";
  s.addShape(pres.shapes.LINE, o);
}
async function iconCircle(s, Comp, x, y, d, fillScheme, name) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fillScheme }, line: { color: fillScheme, width: 0 }, objectName: (name || "icon") + " circle" });
  s.addImage({ data: await icon(Comp, "FFFFFF"), x: x + d * 0.24, y: y + d * 0.24, w: d * 0.52, h: d * 0.52, altText: name || "icon" });
}
const chartBase = (o) => Object.assign({
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  catAxisLabelColor: HEX.ink, valAxisLabelColor: HEX.grey, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  valGridLine: { color: "E5E7EB", size: 0.75 }, catGridLine: { style: "none" },
  dataLabelFontSize: 11, dataLabelColor: HEX.ink, legendFontSize: 12, legendColor: HEX.ink,
  catAxisLineShow: true, valAxisLineShow: false,
}, o);

// marks which research question(s) a results slide answers
function answers(s, list) {
  const pw = 0.66, g = 0.08, right = 11.05;
  const x0 = right - list.length * pw - (list.length - 1) * g;
  txt(s, "ANSWERS", { x: x0 - 1.0, y: 0.33, w: 0.92, h: 0.32, fontSize: 10, bold: true, color: GREY, align: "right", valign: "middle", charSpacing: 1 });
  list.forEach((r, i) => {
    const x = x0 + i * (pw + g);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 0.33, w: pw, h: 0.32, fill: { color: RED }, line: { color: RED, width: 0 }, rectRadius: 0.06, objectName: r + " answered" });
    txt(s, r, { x, y: 0.33, w: pw, h: 0.32, fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle" });
  });
}
// Keynote re-lays-out native chart titles, so titles are drawn as text boxes above the plot
function addChart(s, type, data, o) {
  if (o.showTitle && o.title) {
    txt(s, o.title, { x: o.x, y: o.y, w: o.w, h: 0.32, fontSize: Math.min(o.titleFontSize || 12, 13), bold: true, color: INK, align: "center", valign: "middle" });
    o.y += 0.36; o.h -= 0.36;
  }
  o.showTitle = false; delete o.title;
  s.addChart(type, data, o);
}
// =====================================================================
// 14 slides: title, 3 motivation, 3 methodology, 5 results, conclusion, closing.
// The footer tracker carries the agenda, so there are no divider slides.
(async () => {
  const STR_DARK = "9A6700";
  // ---------------- 1. TITLE ----------------
  sec("Opening");
  {
    const s = pres.addSlide({ masterName: "TITLE", sectionTitle: section });
    // graph motif on the right: a small heterogeneous graph in translucent white
    const N = [[10.0, 1.7, 0.34], [11.6, 1.35, 0.22], [12.4, 2.6, 0.3], [10.9, 2.85, 0.5], [9.5, 3.55, 0.26], [11.9, 3.9, 0.24],
               [10.6, 4.6, 0.32], [12.5, 4.95, 0.2], [9.7, 5.45, 0.22], [11.4, 5.75, 0.28]];
    const E = [[0, 3], [1, 3], [2, 3], [3, 4], [3, 5], [5, 2], [3, 6], [6, 4], [6, 7], [5, 7], [6, 8], [6, 9], [9, 7], [0, 1]];
    for (const [a, b] of E) {
      const [x1, y1] = N[a], [x2, y2] = N[b];
      const o = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001, line: { color: WHITE, width: 1.25, transparency: 55 }, objectName: "motif edge" };
      if ((x2 < x1) !== (y2 < y1)) o.flipH = true;
      s.addShape(pres.shapes.LINE, o);
    }
    N.forEach(([x, y, d], i) => s.addShape(pres.shapes.OVAL, { x: x - d / 2, y: y - d / 2, w: d, h: d, fill: { color: WHITE, transparency: i === 3 ? 0 : 35 }, line: { color: WHITE, width: 0 }, objectName: "motif node" }));
    s.addImage({ path: A + "logo_white.png", x: MX + 0.2, y: 0.55, w: 2.3, h: 0.70, altText: "Universität Koblenz logo" });
    // stacked blocks with fixed gaps so a wider substitute font cannot push one into the next
    txt(s, "MASTER'S THESIS DEFENCE", { x: MX + 0.2, y: 1.7, w: 7, h: 0.35, fontSize: 14, bold: true, color: WHITE, charSpacing: 3 });
    txt(s, "Threat Detection Using Heterogeneous Graph Neural Networks with Semantic and Temporal Features",
      { x: MX + 0.2, y: 2.15, w: 8.4, h: 2.05, fontSize: 28, bold: true, color: WHITE, fontFace: "Georgia", valign: "top" });
    s.addShape(pres.shapes.LINE, { x: MX + 0.2, y: 4.45, w: 2.2, h: 0, line: { color: WHITE, width: 2 }, objectName: "title rule" });
    txt(s, "Kandenghat Abhishek Menon", { x: MX + 0.2, y: 4.7, w: 8.3, h: 0.45, fontSize: 22, bold: true, color: WHITE, valign: "middle" });
    txt(s, "M.Sc. Web and Data Science  ·  Universität Koblenz", { x: MX + 0.2, y: 5.18, w: 8.3, h: 0.35, fontSize: 15, color: WHITE, valign: "middle" });
    txt(s, "First supervisor:  Prof. Dr. Andreas Mauthe", { x: MX + 0.2, y: 5.95, w: 8.3, h: 0.32, fontSize: 14, color: WHITE, valign: "middle" });
    txt(s, "Second supervisor:  Christopher Latz", { x: MX + 0.2, y: 6.3, w: 8.3, h: 0.32, fontSize: 14, color: WHITE, valign: "middle" });
    s.addNotes(
      "[~30 s] Good morning. My thesis is about detecting malware from what it does at runtime rather than what it looks like, " +
      "and doing that in a way that stays robust when an attacker deliberately disguises the behaviour. " +
      "Four parts, shown along the bottom of every slide: motivation and the gap, the methodology, the results, which take most of the time, and the conclusion.");
  }

  // ---------------- 2. MOTIVATION ----------------
  sec("Motivation & Gap");
  {
    const s = content("01 · Motivation", "Malware changes its form, not its intent",
      "[~70 s] Malware does not need to keep a fixed appearance to keep its intent. Refactoring API calls, repacking and reordering change how a sample looks while it still does the same thing. " +
      "That is why the field moved from signatures to behaviour. But behavioural detectors usually read one view of behaviour, and each view has a blind spot. " +
      "A model that reads only structure is left guessing between a benign backup and a ransomware encryption phase: both write to many files and call crypto APIs; what separates them is pace. " +
      "A model that reads only literal call names is defeated by API substitution. A model over a flat sequence loses the cross-process topology of a multi-process loader. " +
      "The point is the last line: a detector is only as strong as the dimension it leaves unmodelled, because the adversary simply attacks that one.", 0);
    txt(s, "Refactoring, repacking and reordering change how a sample looks while it still does the same thing. A detector that reads one view of behaviour can be evaded through that view.",
      { x: MX, y: 1.6, w: 12.1, h: 0.7, fontSize: 16, color: INK });
    const cards = [
      ["Reads only structure", "A benign backup and a ransomware encryption phase both write many files and call cryptographic APIs. What separates them is largely pace.", "Blind to timing", TMP, fa.FaProjectDiagram],
      ["Reads only call names", "API substitution swaps a call for a functional equivalent. The behaviour is identical; only the symbols change.", "Blind to meaning", SEM, fa.FaFont],
      ["Reads a flat sequence", "A multi-process loader spreads its work across processes. One ordered stream loses the cross-process topology.", "Blind to structure", STR, fa.FaStream],
    ];
    const w = 3.83, g = 0.3;
    for (let i = 0; i < 3; i++) {
      const [h, b, tag, col, ic] = cards[i];
      const x = MX + i * (w + g), y = 2.5;
      card(s, x, y, w, 3.05, LIGHT, h);
      await iconCircle(s, ic, x + 0.3, y + 0.3, 0.62, GREY, h);
      txt(s, h, { x: x + 1.1, y: y + 0.3, w: w - 1.3, h: 0.62, fontSize: 18, bold: true, valign: "middle" });
      txt(s, b, { x: x + 0.3, y: y + 1.1, w: w - 0.6, h: 1.25, fontSize: 14, color: INK });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.3, y: y + 2.42, w: 1.95, h: 0.38, fill: { color: col }, line: { color: col, width: 0 }, rectRadius: 0.19, objectName: tag + " pill" });
      txt(s, tag, { x: x + 0.3, y: y + 2.42, w: 1.95, h: 0.38, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle" });
    }
    card(s, MX, 5.85, 12.13, 0.85, HEX.redTint, "takeaway");
    txt(s, "A detector is only as strong as the dimension it leaves unmodelled: the adversary attacks that one.",
      { x: MX + 0.35, y: 5.85, w: 11.5, h: 0.85, fontSize: 17, italic: true, bold: true, color: RED, valign: "middle", fontFace: "Georgia" });
  }

  // ---------------- 3. RELATED WORK & GAP ----------------
  {
    const s = content("01 · Related work & research gap", "Each line of work models only part of behaviour",
      "[~70 s] I grouped the literature into families; this is Table 2.1 of the thesis. Static ML over counts; sequential models; graph neural networks; heterogeneous and temporal graph networks; semantic and LLM-based approaches. " +
      "Each supplies some dimensions and none supplies all. The closest work is GraphShield: it does use a BERT encoder, but only to group call names and shrink the vocabulary. The meaning of a call never reaches the classifier, and their own limitations section lists semantic node labels and joint structural-temporal modelling as future work. " +
      "So the gap: no existing framework carries pace, meaning and typed relations into the decision it makes. Attribution is the secondary gap: post-hoc attention shows where a model looked, not which dimension decided.", 0);
    const Y = "✓", L = "limited", P = "partial", X = "✕";
    const head = ["Approach", "Semantic", "Structural", "Temporal", "Heterog.", "Attribution"].map((t, i) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: GREY }, align: i ? "center" : "left" } }));
    const rows = [
      ["Signature-based", X, X, X, X, Y], ["Static ML", X, L, X, X, P], ["Sequential DL", L, X, Y, X, X],
      ["GraphShield", L, Y, L, X, P], ["HetGNN", X, Y, X, Y, X], ["TGN", X, Y, Y, X, X], ["BEACON (LLM)", Y, X, X, X, P],
    ];
    const cell = (v, i) => ({ text: v, options: { align: i ? "center" : "left", color: v === X ? HEX.base : (v === Y ? INK : GREY), italic: v === L || v === P, bold: v === Y } });
    const body = rows.map(r => r.map(cell));
    const ours = ["This thesis", Y, Y, Y, Y, "gate-based"].map((v, i) => ({ text: v, options: { bold: true, color: RED, fill: { color: HEX.redTint }, align: i ? "center" : "left" } }));
    s.addTable([head, ...body, ours], {
      x: MX, y: 1.62, w: 7.85, colW: [2.05, 1.12, 1.18, 1.12, 1.1, 1.28], rowH: 0.43, fontSize: 12, fontFace: "Arial",
      border: { type: "solid", pt: 0.75, color: HEX.line }, valign: "middle", margin: [0.04, 0.09, 0.04, 0.09],
    });
    txt(s, "Table 2.1 of the thesis.  ✓ first-class   ·   limited / partial   ·   ✕ absent by construction",
      { x: MX, y: 5.65, w: 7.85, h: 0.3, fontSize: 11, color: GREY });
    card(s, 8.8, 1.62, 3.93, 4.33, LIGHT, "gap card");
    txt(s, "The gap", { x: 9.1, y: 1.85, w: 3.4, h: 0.4, fontSize: 20, bold: true, color: RED, fontFace: "Georgia" });
    txt(s, "No existing framework carries all three dimensions into the decision it makes: the pace of execution, the meaning of each call, and the typed relations between entities.",
      { x: 9.1, y: 2.35, w: 3.4, h: 1.9, fontSize: 15 });
    txt(s, "Closest prior work, GraphShield (Amer et al., 2026), uses BERT only to sort call names into groups before training. Its classifier sees the groups, not what each call means.",
      { x: 9.1, y: 4.35, w: 3.4, h: 1.3, fontSize: 13, italic: true, color: GREY });
    txt(s, "Secondary gaps: attribution across dimensions, and no ablation separating how much a dimension is used from whether the decision survives without it.",
      { x: MX, y: 6.15, w: 12.13, h: 0.6, fontSize: 14, color: INK });
  }

  // ---------------- 4. RQs ----------------
  {
    const s = content("01 · Research questions", "Five questions, each answered by one experiment",
      "[~45 s] From the gap come five research questions. RQ1: do pretrained semantic embeddings generalise to obfuscated or unseen behaviour. RQ2: do the three views carry complementary information. " +
      "RQ3: how does the framework compare with single-paradigm baselines, clean and under attack. RQ4: can a decision be attributed. RQ5: does detection hold on threats from an independent source. " +
      "On the right is the experiment that answers each; every one comes back in the results.", 0);
    const rq = [
      ["RQ1", "Semantic generalisation", "Do pretrained semantic embeddings help detection generalise to obfuscated or previously unseen behaviour?", "Embedding substitution"],
      ["RQ2", "Complementarity", "Do meaning, structure and timing carry complementary information, and does modelling them together achieve what none does alone?", "Gate analysis + ablation"],
      ["RQ3", "End-to-end performance", "How does the framework compare with single-paradigm baselines, on clean traces and under attack?", "4 baselines, 4 attacks"],
      ["RQ4", "Interpretability", "Can a decision be attributed to the dimension and the behaviour that produced it?", "Gate weights + saliency"],
      ["RQ5", "Cross-source generalisation", "Does detection hold on threats drawn from an independent source?", "110 unseen threats"],
    ];
    rq.forEach(([tag, t, q, e], i) => {
      const y = 1.62 + i * 1.02;
      card(s, MX, y, 12.13, 0.9, LIGHT, tag + " row");
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.18, y: y + 0.2, w: 0.85, h: 0.5, fill: { color: RED }, line: { color: RED, width: 0 }, rectRadius: 0.08, objectName: tag + " tag" });
      txt(s, tag, { x: MX + 0.18, y: y + 0.2, w: 0.85, h: 0.5, fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle" });
      txt(s, t, { x: MX + 1.25, y, w: 2.75, h: 0.9, fontSize: 16, bold: true, valign: "middle" });
      txt(s, q, { x: MX + 4.05, y, w: 5.75, h: 0.9, fontSize: 14, valign: "middle" });
      txt(s, "→ " + e, { x: MX + 9.95, y, w: 2.05, h: 0.9, fontSize: 13, bold: true, color: GREY, valign: "middle" });
    });
  }

  // ---------------- 5. DATA + REPRESENTATION ----------------
  sec("Methodology");
  {
    const s = content("02 · Methodology · data and representation", "Each execution trace becomes one typed behavioural graph",
      "[~80 s] The corpus is 9,333 behavioural graphs built from sandbox execution traces: ransomware, other malware and benign software, split 70/15/15, stratified. " +
      "The primary source is the Carpenter and Luo multi-stage malware dataset from the CAPE sandbox. I supplemented it from Hybrid Analysis: mainly benign software, 1,987 benign samples across seven categories, " +
      "and 110 recent threats that come through a different sandbox and acquisition route. Those 110 never touch training; they are the RQ5 test. Graphs range from ten to over 1.5 million API nodes, which is why large graphs are compressed and capped. " +
      "Each trace becomes a heterogeneous graph with five node types and five relation types. Every monitored API call is a node; consecutive calls are joined by temporal edges, the spine of the graph. " +
      "Processes, files, registry keys and network endpoints hang off the calls that touch them. Each API node carries the 768-dimensional MPNet embedding of its name plus seven metadata features, so meaning stays on the node and is never collapsed into a vocabulary, which is exactly what GraphShield does differently. " +
      "And each temporal edge carries 14 timing features: pacing, bursts, regularity, timeline position and process context. " +
      "Each of the five relation types has its own hybrid GAT plus SAGE operator with its own weights, two layers, and the results are summed at each node. Inside each operator a gate g, between 0 and 1 for every node and every feature, mixes the two: g times the attention output plus one minus g times the SAGE output, and it reads both outputs and how much they disagree to set g. That is why structural message passing holds 87 percent of the 16.1 million parameters: GAT 3.76 million, SAGE 3.73 million, and the gate between them 6.50 million.", 1);
    const tiles = [
      ["9,333", "graphs from sandbox execution traces"],
      ["3", "classes: ransomware · other malware · benign"],
      ["70 / 15 / 15", "stratified train / validation / test"],
      ["1,987", "benign samples in 7 software categories"],
      ["10 → 1.5 M", "API nodes per graph, before capping"],
      ["110", "unseen-source threats, never trained on"],
    ];
    const tw = (12.13 - 5 * 0.15) / 6;
    tiles.forEach(([n, l], i) => {
      const x = MX + i * (tw + 0.15), y = 1.55, hot = i === 5;
      card(s, x, y, tw, 1.15, hot ? RED : LIGHT, "stat " + n);
      txt(s, n, { x: x + 0.15, y: y + 0.1, w: tw - 0.3, h: 0.5, fontSize: 21, bold: true, color: hot ? WHITE : RED, fontFace: "Georgia", valign: "middle" });
      txt(s, l, { x: x + 0.15, y: y + 0.6, w: tw - 0.3, h: 0.5, fontSize: 11, color: hot ? WHITE : INK });
    });
    // graph diagram
    card(s, MX, 2.85, 6.75, 3.9, LIGHT, "graph diagram panel");
    const api = [1.15, 2.65, 4.15, 5.65], ay = 4.2, d = 0.65;
    const px = 2.95, py = 3.05, pw = 1.85, ph = 0.5;
    for (const x of api) line(s, px + pw / 2, py + ph, x + d / 2, ay, HEX.str, 1.5, false, "executes edge");
    const rw = 1.3, ry = 5.5, rh = 0.48;
    line(s, api[0] + d / 2, ay + d, 1.625 + rw / 2, ry, HEX.grey, 1.25, false, "operates_on edge");
    line(s, api[1] + d / 2, ay + d, 1.625 + rw / 2, ry, HEX.grey, 1.25, false, "operates_on edge");
    line(s, api[2] + d / 2, ay + d, 3.875 + rw / 2, ry, HEX.grey, 1.25, false, "modifies edge");
    line(s, api[3] + d / 2, ay + d, 5.375 + rw / 2, ry, HEX.grey, 1.25, false, "connects_to edge");
    for (let i = 0; i < 3; i++) line(s, api[i] + d, ay + d / 2, api[i + 1], ay + d / 2, HEX.tmp, 2.25, true, "temporal edge");
    api.forEach((x, i) => {
      s.addShape(pres.shapes.OVAL, { x, y: ay, w: d, h: d, fill: { color: SEM }, line: { color: WHITE, width: 1.5 }, objectName: "api node " + (i + 1) });
      txt(s, "api", { x, y: ay, w: d, h: d, fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle" });
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: px, y: py, w: pw, h: ph, fill: { color: STR }, line: { color: STR, width: 0 }, rectRadius: 0.1, objectName: "process node" });
    txt(s, "process", { x: px, y: py, w: pw, h: ph, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle" });
    [["file", 1.625], ["registry", 3.875], ["network", 5.375]].forEach(([n, x]) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: ry, w: rw, h: rh, fill: { color: WHITE }, line: { color: GREY, width: 1.25 }, rectRadius: 0.1, objectName: n + " node" });
      txt(s, n, { x, y: ry, w: rw, h: rh, fontSize: 13, bold: true, color: GREY, align: "center", valign: "middle" });
    });
    txt(s, "temporal", { x: 4.8, y: ay + d + 0.04, w: 0.85, h: 0.28, fontSize: 11, bold: true, color: TMP, align: "center" });
    txt(s, "executes", { x: 1.6, y: py + 0.1, w: 1.25, h: 0.3, fontSize: 11, bold: true, color: STR, align: "right" });
    txt(s, "operates_on · modifies · connects_to", { x: 1.0, y: 6.15, w: 6.0, h: 0.3, fontSize: 11, bold: true, color: GREY });
    // feature cards
    const fc = [
      ["API node · 775-d", "768-d MPNet embedding of the call name + 7 metadata features. Meaning stays on the node, never collapsed into a smaller vocabulary.", SEM, HEX.semTint, fa.FaFont],
      ["Temporal edge · 14 features", "Pacing, bursts, local regularity, timeline position and process context between consecutive calls.", TMP, HEX.tmpTint, fa.FaStopwatch],
      ["5 node types · 5 typed relations", "Each relation has its own hybrid GAT + SAGE operator, 2 layers: a file write is not a process spawn. This holds 87 % of the 16.1 M parameters.", STR, HEX.strTint, fa.FaProjectDiagram],
    ];
    const cy = [2.85, 4.05, 5.15], ch = [1.1, 0.98, 1.6];
    for (let i = 0; i < 3; i++) {
      const [t, b, col, tint, ic] = fc[i];
      const y = cy[i];
      card(s, 7.65, y, 5.08, ch[i], tint, t);
      await iconCircle(s, ic, 7.85, y + (i === 2 ? 0.45 : 0.26), 0.58, col, t);
      txt(s, t, { x: 8.62, y: y + 0.1, w: 3.95, h: 0.35, fontSize: 15, bold: true, color: INK });
      txt(s, b, { x: 8.62, y: y + 0.45, w: 3.95, h: i === 1 ? 0.45 : 0.62, fontSize: 12, color: INK });
    }
    // hybrid gate formula (Eq. 4.1-4.2) under the relation card text
    txt(s, [
      { text: "h = g ⊙ h", options: { bold: true, fontFace: "Cambria", fontSize: 13 } },
      { text: "GAT", options: { bold: true, fontFace: "Cambria", fontSize: 13, subscript: true } },
      { text: " + (1 − g) ⊙ h", options: { bold: true, fontFace: "Cambria", fontSize: 13 } },
      { text: "SAGE", options: { bold: true, fontFace: "Cambria", fontSize: 13, subscript: true, breakLine: true } },
      { text: "g = σ(MLP[GAT ‖ SAGE ‖ 2·|GAT − SAGE|]) · Eq. 4.1–4.2", options: { fontSize: 10, color: GREY } },
    ], { x: 8.62, y: 5.15 + 1.03, w: 3.95, h: 0.5, color: STR_DARK, paraSpaceAfter: 1, margin: 0 });
  }

  // ---------------- 6. ARCHITECTURE ----------------
  {
    const s = content("02 · Methodology · architecture", "One model reads the graph along three branches",
      "[~70 s] The HeteroTemporalGNN is a single end-to-end model of about 16.1 million parameters. Each node type is first projected into its own hidden space. " +
      "From there three branches read the graph at different depths. The semantic branch is the mean of the projected API states before any message passing: meaning alone. " +
      "The structural branch is two layers of hybrid GAT-SAGE message passing over the typed relations. The temporal branch runs a bidirectional LSTM over the message-passed API states in time order, with six-head attention pooling. " +
      "An adaptive gate weighs the three branch embeddings for every sample, then a shared trunk with behavioural statistics feeds three class heads. It is trained on three classes and evaluated as threat versus benign. " +
      "Follow one sample along the numbers: each call arrives as 775 numbers and each step as 14 timing numbers; every node is projected into its own space; the three branches each condense the sample into 128 numbers; the gate weighs them, for example 0.66, 0.15 and 0.19, and adds them into one vector; the trunk adds 15 volume statistics and the heads give three class probabilities, and the highest decides.", 1);
    const box = (x, y, w, h, title, sub, fill, fg, name, flow, flowColor) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.1, objectName: name, shadow: shadow() });
      const runs = [{ text: title, options: { bold: true, fontSize: 15, breakLine: true } }, { text: sub, options: { fontSize: 12, breakLine: !!flow } }];
      if (flow) runs.push({ text: flow, options: { fontSize: 11, bold: true, italic: true, color: flowColor } });
      txt(s, runs, { x: x + 0.12, y, w: w - 0.24, h, color: fg, valign: "middle", align: "center", paraSpaceAfter: 4 });
    };
    // one sample's journey, written into each box (flow line, bold italic)
    const yS = 1.45, yT = 2.95, yM = 4.45, bh = 1.38, c = (y) => y + bh / 2;
    box(MX, 2.6, 1.85, 2.05, "Enriched graph", "typed nodes, timed edges", LIGHT, INK, "input box", "① each call = 775 numbers, each step = 14 timing numbers", RED);
    box(2.75, 2.6, 1.85, 2.05, "Type-specific projections", "api → 128-d, resource → 64-d", LIGHT, INK, "projection box", "② every node now in its own hidden space", RED);
    box(5.05, yS, 3.4, bh, "Semantic branch", "mean of projected API states, before message passing", HEX.semTint, INK, "semantic branch", "③ what the calls mean → z_sem (128)", HEX.sem);
    box(5.05, yT, 3.4, bh, "Structural branch", "hybrid GAT–SAGE message passing, 2 layers, 6 heads", HEX.strTint, INK, "structural branch", "③ process → call → file links → z_str (128)", "9A6700");
    box(5.05, yM, 3.4, bh, "Temporal branch", "BiLSTM over message-passed calls in time order, 6-head attention pooling", HEX.tmpTint, INK, "temporal branch", "③ calls read in time order → z_tmp (128)", HEX.tmp);
    box(8.9, 2.35, 2.0, 2.55, "Adaptive gate", "one weight per branch, set for every sample", RED, WHITE, "gate box", "④ e.g. w = 0.66 / 0.15 / 0.19 → z = Σ w·z (128)", "FFE3E5");
    box(11.2, 2.35, 1.53, 2.55, "Trunk + 3 heads", "→ threat vs benign", INK, WHITE, "output box", "⑤ z + 15 stats → 3 class probabilities → argmax", "D6DAE2");
    line(s, 2.45, 3.625, 2.75, 3.625, HEX.grey, 1.75, true, "arrow");
    line(s, 4.6, 3.625, 4.82, 3.625, HEX.grey, 1.75, false, "arrow");
    line(s, 4.82, c(yS), 4.82, c(yM), HEX.grey, 1.75, false, "bus");
    for (const y of [c(yS), c(yT), c(yM)]) line(s, 4.82, y, 5.05, y, HEX.grey, 1.75, true, "arrow");
    for (const y of [c(yS), c(yT), c(yM)]) line(s, 8.45, y, 8.67, y, HEX.grey, 1.75, false, "arrow");
    line(s, 8.67, c(yS), 8.67, c(yM), HEX.grey, 1.75, false, "bus");
    line(s, 8.67, 3.63, 8.9, 3.63, HEX.grey, 1.75, true, "arrow");
    line(s, 10.9, 3.625, 11.2, 3.625, HEX.grey, 1.75, true, "arrow");
    txt(s, [
      { text: "≈ 16.1 M parameters", options: { bold: true } }, { text: "   ·   one end-to-end model   ·   trained on 3 classes, evaluated as threat vs benign   ·   per-type behavioural statistics join after fusion" },
    ], { x: MX, y: 6.05, w: 12.13, h: 0.5, fontSize: 13, color: GREY });
    txt(s, "①–⑤ follow one sample through the model · gate weights are an example", { x: MX, y: 6.55, w: 12.13, h: 0.3, fontSize: 11, italic: true, color: GREY });
  }

  // ---------------- 7. WHAT IS NEW + THREE-CLASS TRAINING ----------------
  {
    const s = content("02 · Methodology · design decisions", "Two learned balances, and three-class training",
      "[~80 s] Two design decisions carry the contribution. First, the hybrid aggregator. Attention finds the single decisive edge, a process spawn or a persistence write among thousands of routine reads, but its weights are least reliable on unfamiliar neighbourhoods. " +
      "GraphSAGE generalises to unseen entities but weighs all neighbours equally. I modified SAGE to consume edge features, and a gate reads where the two operators disagree and decides per node and per feature which to trust. " +
      "Second, the fusion gate. The dimension that decides a case varies: one sample gives itself away by what it calls, another by how fast it acts. So the gate sets the balance for every sample, and the same weights are read out as attribution. " +
      "Third, a training decision. The model is trained on three classes but evaluated as threat versus benign. Ransomware has its own temporal signature, encryption bursts and rapid mass file modification, which one merged threat class would dilute; separating the classes forces the model to learn what tells the threats apart, which matters most for threats it has never seen. " +
      "The deployed question is binary, so either malicious class counts as a threat. And the errors fall where the design intends: all 21 misclassified ransomware samples were called other malware, none benign. " +
      "Training uses focal loss with oversampling, and a checkpoint is only kept if ransomware and benign recall clear minimum floors.", 1);
    const cols = [
      ["Hybrid relational aggregator", fa.FaCodeBranch, STR, HEX.strTint, [
        "Attention finds the one decisive edge, but is least reliable on unfamiliar neighbourhoods.",
        "Edge-aware GraphSAGE generalises to unseen entities, but weighs all neighbours equally.",
        "A gate decides, per node and per feature, which to trust.",
      ], "g = σ(MLP[h_GAT, h_SAGE, λ·|h_GAT − h_SAGE|])"],
      ["Adaptive three-branch fusion", fa.FaSlidersH, RED, HEX.redTint, [
        "The deciding dimension varies: meaning for one trace, pacing for another.",
        "The gate weighs the three branches for every sample, never one fixed blend.",
        "Its weights are read out as attribution, with no post-hoc model.",
      ], "w = softmax(MLP[z_sem, z_str, z_tmp])"],
      ["Three-class training, binary verdict", fa.FaLayerGroup, GREY, LIGHT, [
        "Ransomware's timing signature (encryption bursts, mass file changes) would be diluted in one merged threat class.",
        "Three classes force the model to learn what tells the threats apart, which matters most for unseen threats.",
        "All 21 misclassified ransomware go to other malware, none to benign.",
      ], "threat ⇔ ŷ ∈ {ransomware, other malware}  (Eq. 4.8)"],
    ];
    const w = 3.88, g = 0.245;
    for (let i = 0; i < 3; i++) {
      const [t, ic, col, tint, bullets, f] = cols[i];
      const x = MX + i * (w + g);
      card(s, x, 1.55, w, 4.45, tint, t);
      await iconCircle(s, ic, x + 0.25, 1.75, 0.58, col, t);
      txt(s, t, { x: x + 0.95, y: 1.7, w: w - 1.1, h: 0.7, fontSize: 15, bold: true, valign: "middle", fontFace: "Georgia" });
      txt(s, bullets.map((b, k) => ({ text: b, options: { bullet: { indent: 14 }, breakLine: k < bullets.length - 1 } })),
        { x: x + 0.25, y: 2.55, w: w - 0.45, h: 2.75, fontSize: 13.5, paraSpaceAfter: 8 });
      txt(s, f, { x: x + 0.1, y: 5.35, w: w - 0.2, h: 0.5, fontSize: 12, italic: true, color: INK, fontFace: "Georgia", align: "center", valign: "middle" });
    }
    txt(s, [
      { text: "Training:  ", options: { bold: true } },
      { text: "focal loss (γ = 2) with class-weighted oversampling (9.0 / 3.5 / 1.2); a checkpoint is kept only if validation ransomware recall ≥ 0.40 and benign recall ≥ 0.52." },
    ], { x: MX, y: 6.15, w: 12.13, h: 0.6, fontSize: 13, color: GREY });
  }

  // ---------------- 8. DETECTION + GATE USAGE ----------------
  sec("Results");
  {
    const s = content("03 · Results · detection and branch usage (RQ2)", "No threat missed, and the gate adapts to each class",
      "[~90 s] On the 1,384-sample test split, every one of the 1,085 threats is assigned to the threat side: zero missed. The cost is 16 of 299 benign samples flagged, a false-positive rate of 5.4 percent; every error falls on the conservative side. " +
      "In the three-class view ransomware is the hardest class: recall 0.462 on 39 samples. But all 21 misses are called other malware, none benign: an error in naming the threat, never in detecting it. With 39 samples that recall has a wide interval, which is why threat recall is the headline metric. " +
      "On the right, the first half of RQ2: does the model really use all three branches? I read the gate on 440 samples. Overall it is nearly balanced, 0.35, 0.35, 0.29, so no branch is decorative. " +
      "Per class it adapts: for ransomware the gate puts 0.661 on semantics, which leads 98.7 percent of ransomware decisions; for benign software it shifts to structure, 0.450, leading 78.5 percent. The balance follows the evidence a sample presents.", 2);
    answers(s, ["RQ2", "RQ4"]);
    card(s, MX, 1.62, 5.85, 1.45, HEX.redTint, "hero stat");
    txt(s, "1,085 / 1,085", { x: MX + 0.3, y: 1.68, w: 5.3, h: 0.85, fontSize: 44, bold: true, color: RED, fontFace: "Georgia", valign: "middle" });
    txt(s, "test threats detected: zero missed (threat recall 1.000)", { x: MX + 0.3, y: 2.55, w: 5.3, h: 0.4, fontSize: 14, bold: true });
    [["0.993", "threat F1-score"], ["5.4 %", "false-positive rate (16 of 299 benign)"]].forEach(([n, l], i) => {
      const x = MX + i * 3.0;
      card(s, x, 3.2, 2.85, 0.95, LIGHT, "stat " + n);
      txt(s, n, { x: x + 0.2, y: 3.2, w: 1.2, h: 0.95, fontSize: 24, bold: true, color: INK, fontFace: "Georgia", valign: "middle" });
      txt(s, l, { x: x + 1.45, y: 3.2, w: 1.3, h: 0.95, fontSize: 12, color: GREY, valign: "middle" });
    });
    const hc = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: GREY }, align: "center" } });
    const rc = (t) => ({ text: t, options: { bold: true, align: "left" } });
    const zero = { text: "0", options: { bold: true, color: TMP, fill: { color: HEX.tmpTint }, align: "center" } };
    const n = (t, o) => ({ text: t, options: Object.assign({ align: "center" }, o || {}) });
    s.addTable([
      [hc("Actual \\ predicted"), hc("Ransomware"), hc("Benign"), hc("Other malware")],
      [rc("Ransomware (39)"), n("18"), zero, n("21", { bold: true, color: STR, fill: { color: HEX.strTint } })],
      [rc("Benign (299)"), n("0"), n("283"), n("16")],
      [rc("Other malware (1,046)"), n("6"), zero, n("1,040")],
    ], { x: MX, y: 4.3, w: 5.85, colW: [2.05, 1.25, 1.1, 1.45], rowH: 0.36, fontSize: 12, fontFace: "Arial", border: { type: "solid", pt: 0.75, color: HEX.line }, valign: "middle", margin: [0.03, 0.08, 0.03, 0.08] });
    txt(s, [
      { text: "Ransomware recall 0.462 (n = 39): ", options: { bold: true, color: STR } },
      { text: "all 21 misses go to other malware, none to benign: a naming error, not a missed threat." },
    ], { x: MX, y: 5.88, w: 5.85, h: 0.8, fontSize: 12 });
    const lab = ["Ransomware (75)", "Benign (200)", "Other malware (165)", "All (440)"];
    addChart(s, pres.charts.BAR, [
      { name: "Semantic", labels: lab, values: [0.661, 0.223, 0.374, 0.354] },
      { name: "Structural", labels: lab, values: [0.130, 0.450, 0.337, 0.353] },
      { name: "Temporal", labels: lab, values: [0.209, 0.327, 0.289, 0.293] },
    ], chartBase({
      x: 6.75, y: 1.55, w: 5.98, h: 3.75, barDir: "col", barGrouping: "stacked", chartColors: [HEX.sem, HEX.str, HEX.tmp],
      showValue: true, dataLabelPosition: "ctr", dataLabelColor: "FFFFFF", dataLabelFormatCode: "0.00", dataLabelFontBold: true, dataLabelFontSize: 10,
      valAxisMaxVal: 1, valAxisMinVal: 0, valAxisLabelFormatCode: "0.0", showLegend: true, legendPos: "b", legendFontSize: 11, barGapWidthPct: 55, catAxisLabelFontSize: 11,
      showTitle: true, title: "Mean gate weight by true class (RQ2: are all three used?)", titleFontSize: 13, titleColor: HEX.ink,
    }));
    [["Ransomware → semantic", "0.661", "leads 98.7 % of ransomware decisions", SEM, HEX.semTint],
     ["Benign → structural", "0.450", "leads 78.5 % of benign decisions", STR, HEX.strTint]].forEach(([h, v, l, col, tint], i) => {
      const x = 6.75 + i * 3.06;
      card(s, x, 5.45, 2.92, 1.3, tint, h);
      txt(s, h, { x: x + 0.2, y: 5.53, w: 2.6, h: 0.3, fontSize: 13, bold: true, color: col });
      txt(s, v, { x: x + 0.2, y: 5.85, w: 1.0, h: 0.75, fontSize: 24, bold: true, color: INK, fontFace: "Georgia", valign: "middle" });
      txt(s, l, { x: x + 1.2, y: 5.85, w: 1.6, h: 0.75, fontSize: 12, color: GREY, valign: "middle" });
    });
  }

  // ---------------- 9. ABLATION (RQ2) + SUBSTITUTION (RQ1) ----------------
  {
    const s = content("03 · Results · ablation (RQ2) and semantic robustness (RQ1)", "Detection survives losing any one branch",
      "[~110 s] Second half of RQ2: what survives when a branch is removed outright? 300 test graphs, 237 threats. The surprising one is semantics. It has the largest contribution share, yet removing it misses not a single threat; its loss shows up as exact-class accuracy falling from 285 to 228. " +
      "Removing structure is the only single removal that misses threats, ten, and the finer ablation traces all ten to one relation, process to API. So contribution and indispensability are different properties: the branches overlap on whether a sample is a threat and specialise on which threat. Only the full model reaches 285 of 300; no single branch exceeds 279. " +
      "On the right, RQ1: has the semantic branch learned what calls mean, or memorised embedding identities? Switching it off cannot tell the two apart; substitution can. In the intact model, swapping almost every embedding barely moves anything: no over-dependence. " +
      "So I silenced structure, forcing reliance on semantics and timing, and replaced every embedding two ways. With a functional neighbour, meaning kept: missed threats go from 8 to 10 of 226, 4.4 percent, and 0.7 percent of all 300 predictions change. With a random operation, meaning destroyed: 21 of 226 missed, 9.3 percent, and 5.3 percent of predictions change. " +
      "A memorising branch would be hurt equally by both. The gap shows it reads what a call does.", 2);
    answers(s, ["RQ1", "RQ2"]);
    const head = (x, w, tag, t) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.58, w: 0.75, h: 0.36, fill: { color: RED }, line: { color: RED, width: 0 }, rectRadius: 0.06, objectName: tag + " tag" });
      txt(s, tag, { x, y: 1.58, w: 0.75, h: 0.36, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle" });
      txt(s, t, { x: x + 0.9, y: 1.58, w: w - 0.9, h: 0.36, fontSize: 15, bold: true, valign: "middle" });
    };
    // left: ablation
    head(MX, 6.0, "RQ2", "Remove one branch entirely");
    s.addImage({ path: A + "branch_removal.png", x: MX, y: 2.05, w: 6.0, h: 6.0 / 2.1, altText: "Thesis Figure 5.4: threats missed and exact-class accuracy across the seven branch configurations" });
    txt(s, "Thesis Fig. 5.4 · 300 test graphs (237 threats, 63 benign)", { x: MX, y: 4.93, w: 6.0, h: 0.25, fontSize: 10, color: GREY });
    card(s, MX, 5.22, 6.0, 1.53, HEX.redTint, "ablation insight");
    txt(s, [
      { text: "Contribution ≠ indispensability", options: { bold: true, color: RED, fontSize: 14, breakLine: true } },
      { text: "No semantic: ", options: { bold: true, color: SEM } }, { text: "0 missed, exact class 285 → 228.  ", options: {} },
      { text: "No temporal: ", options: { bold: true, color: TMP } }, { text: "1 missed.", options: { breakLine: true } },
      { text: "No structural: ", options: { bold: true, color: STR } }, { text: "10 missed (95.8 % recall), all via process → API.", options: { breakLine: true } },
      { text: "Only the full model reaches 285 / 300 exact class; no single branch exceeds 279." },
    ], { x: MX + 0.2, y: 5.3, w: 5.65, h: 1.4, fontSize: 12, paraSpaceAfter: 2 });
    // right: substitution
    head(6.85, 5.88, "RQ1", "Swap every embedding, structure silenced");
    addChart(s, pres.charts.BAR, [{ name: "Threats missed (of 226)", labels: ["Intact", "No structure", "+ synonym swap", "+ random swap"], values: [0, 8, 10, 21] }],
      chartBase({
        x: 6.85, y: 2.0, w: 5.88, h: 2.65, barDir: "col", chartColors: [HEX.base, HEX.grey, HEX.sem, HEX.red],
        showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelFontBold: true, valAxisMinVal: 0, valAxisMaxVal: 25, valAxisMajorUnit: 5,
        showLegend: false, barGapWidthPct: 60, catAxisLabelFontSize: 11, showTitle: true, title: "Threats missed of 226 (300 test graphs, Table 5.11)", titleFontSize: 12, titleColor: HEX.ink,
      }));
    [["Synonym swap", "meaning kept", "4.4 %", "of 226 threats missed (10) · 0.7 % of predictions change", SEM, HEX.semTint],
     ["Random swap", "meaning destroyed", "9.3 %", "of 226 threats missed (21) · 5.3 % of predictions change", RED, HEX.redTint]].forEach(([h, sub, v, l, col, tint], i) => {
      const x = 6.85 + i * 3.01;
      card(s, x, 4.75, 2.87, 1.0, tint, h);
      txt(s, [{ text: h, options: { bold: true, color: col } }, { text: " · " + sub, options: { color: GREY } }], { x: x + 0.15, y: 4.79, w: 2.6, h: 0.3, fontSize: 11 });
      txt(s, v, { x: x + 0.15, y: 5.09, w: 1.05, h: 0.6, fontSize: 22, bold: true, color: col, fontFace: "Georgia", valign: "middle" });
      txt(s, l, { x: x + 1.2, y: 5.09, w: 1.6, h: 0.6, fontSize: 9.5, valign: "middle" });
    });
    txt(s, [
      { text: "Step 1, intact model (Table 5.10): ", options: { bold: true } },
      { text: "the same swaps barely move it (test: 1 → 2 missed of 306), so there is no over-dependence on semantics.", options: { breakLine: true } },
      { text: "Step 2, structure silenced (above): ", options: { bold: true } },
      { text: "only the meaning-destroying swap hurts; a memorising branch would be hurt by both." },
    ], { x: 6.85, y: 5.85, w: 5.88, h: 0.9, fontSize: 11.5, color: INK, paraSpaceAfter: 2 });
  }

  // ---------------- 10. RQ3 ATTACKS ----------------
  {
    const s = content("03 · Results · baselines under attack (RQ3)", "Every baseline breaks under attack; the framework holds",
      "[~100 s] This is the central result. Four established detectors, each committed to one view: TF-IDF with XGBoost for call identity, a static GAT for structure, an MPNet embedding classifier for meaning, a BiLSTM for order. " +
      "On clean traces they are competitive; XGBoost actually has the best macro-F1. The framework holds no clean-data lead, and I do not claim one. " +
      "Then each gets a behaviour-preserving attack aimed at its own dimension: the malware still does exactly the same thing. Renaming calls erases XGBoost: all 275 threats missed. Perturbing the graph drives GAT to 144 of 275 missed. " +
      "An out-of-context rename makes the semantic MLP miss all 125. The BiLSTM survives reordering, but under renaming it flags all 200 benign traces: it fails by crying wolf. Its zero missed threats there is an artefact: a detector that flags everything catches every threat trivially while distinguishing nothing. " +
      "Under the identical attacks the framework misses at most two threats and never loses more than two points of macro-F1, because an attack on one dimension leaves it two intact views.", 2);
    answers(s, ["RQ3"]);
    txt(s, "Clean traces: no advantage. XGBoost has the best macro-F1 (0.874 vs 0.827); the BiLSTM scores 0.840 vs 0.763 but misses 4 threats where the framework misses none. The separation appears only once traces are manipulated.",
      { x: MX, y: 1.55, w: 12.13, h: 0.6, fontSize: 14, color: GREY });
    const labels = ["Classical (rename)", "Static GNN (structure)", "Semantic (rename)", "Sequential (reorder)"];
    addChart(s, pres.charts.BAR, [
      { name: "Baseline, clean", labels, values: [0.874, 0.800, 0.769, 0.840] },
      { name: "Framework, clean", labels, values: [0.827, 0.827, 0.763, 0.763] },
      { name: "Baseline, attacked", labels, values: [0.198, 0.404, 0.254, 0.728] },
      { name: "Framework, attacked", labels, values: [0.812, 0.823, 0.744, 0.758] },
    ], chartBase({
      x: MX, y: 2.15, w: 7.2, h: 4.25, barDir: "col", barGrouping: "clustered", chartColors: [HEX.base, "EBA3A8", HEX.grey, HEX.red],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 8, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0",
      showLegend: true, legendPos: "b", barGapWidthPct: 30, catAxisLabelFontSize: 11, showTitle: true, title: "Macro-F1, clean (Table 5.12) and attacked (Fig. 5.9)", titleFontSize: 14, titleColor: HEX.ink,
    }));
    txt(s, [
      { text: "Crying wolf: ", options: { bold: true, color: INK } },
      { text: "the Sequential bars show reordering, which the BiLSTM survives (0.728). Under renaming it flags all 200 benign traces and falls to 0.136 (Table 5.13), so its zero missed threats there is no success.", options: {} },
    ], { x: MX, y: 6.42, w: 7.2, h: 0.4, fontSize: 10, italic: true, color: GREY, margin: 0 });
    const hc = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: GREY }, align: "center" } });
    const at = (a, b) => ({ text: a + "\n" + b, options: { align: "left", bold: true } });
    const bad = (t) => ({ text: t, options: { bold: true, color: INK, align: "center" } });
    const good = (t) => ({ text: t, options: { bold: true, color: RED, fill: { color: HEX.redTint }, align: "center" } });
    s.addTable([
      [hc("Baseline · attack"), hc("Baseline"), hc("Framework")],
      [at("XGBoost", "API rename"), bad("275 / 275\nmissed"), good("2 missed")],
      [at("GAT", "graph perturbation"), bad("144 / 275\nmissed"), good("0 missed")],
      [at("Semantic MLP", "out-of-context rename"), bad("125 / 125\nmissed"), good("0 missed")],
      [at("BiLSTM", "API rename"), bad("200 / 200\nbenign flagged"), good("0 missed\n7 flagged")],
    ], { x: 8.05, y: 2.2, w: 4.68, colW: [1.8, 1.5, 1.38], rowH: [0.42, 0.78, 0.78, 0.78, 0.78], fontSize: 12, fontFace: "Arial", border: { type: "solid", pt: 0.75, color: HEX.line }, valign: "middle", margin: [0.04, 0.09, 0.04, 0.09] });
    txt(s, [
      { text: "Note: ", options: { bold: true, color: INK } },
      { text: "under renaming the BiLSTM misses no threat only because it calls every trace a threat (crying wolf): all 200 benign flagged, macro-F1 0.136. A zero bought by flagging everything is a failure, not a success.", options: {} },
    ], { x: 8.05, y: 5.8, w: 4.68, h: 0.55, fontSize: 9.5, italic: true, color: GREY, margin: 0 });
    txt(s, "At most 2 threats missed under any attack, and never more than two points of macro-F1 lost.",
      { x: 8.05, y: 6.38, w: 4.68, h: 0.5, fontSize: 13, bold: true, color: RED, fontFace: "Georgia", margin: 0 });
  }

  // ---------------- 11. RQ5 UNSEEN SOURCE ----------------
  {
    const s = content("03 · Results · cross-source generalisation (RQ5)", "All 110 threats from an unseen source detected",
      "[~60 s] RQ5. Before measuring detection I measured how unfamiliar these 110 threats are, so the result cannot be memorisation. They come from a separate source, withheld from training entirely. " +
      "27.8 percent of their distinct calls never appear in training; the Jensen-Shannon divergence between the call distributions is 0.534; only 6 of the 20 most frequent calls are shared. " +
      "The chart shows where the shift falls: LdrGetProcedureAddress is a fifth of the unseen calls and all but absent from training, while the registry and memory-protection calls that dominate training recede. " +
      "The result: every one returned as a threat, the least confident at 0.865, so none close to the benign boundary, with 6 of 200 familiar benign traces flagged. " +
      "On these threats the gate leans harder on structure, 0.41, and timing, 0.33, while semantics keeps 0.25, still reading meaning from names it never met. This is one genuine shift of source, not breadth across many families.", 2);
    answers(s, ["RQ5"]);
    const st = [["27.8 %", "of the unseen set's distinct calls never occur in training (71 of 255)"], ["0.534", "Jensen–Shannon divergence between the call distributions"],
                ["6 / 20", "most frequent calls shared between the two sets"], ["0", "samples overlapping the training corpus"]];
    st.forEach(([n, l], i) => {
      const x = MX + (i % 2) * 2.95, y = 1.62 + Math.floor(i / 2) * 1.8;
      card(s, x, y, 2.8, 1.65, LIGHT, "stat " + n);
      txt(s, n, { x: x + 0.22, y: y + 0.15, w: 2.4, h: 0.65, fontSize: 28, bold: true, color: INK, fontFace: "Georgia", valign: "middle" });
      txt(s, l, { x: x + 0.22, y: y + 0.85, w: 2.4, h: 0.72, fontSize: 12, color: GREY });
    });
    const calls = ["NtQueryValueKey", "NtProtectVirtualMemory", "GetSystemMetrics", "FindWindowA", "LdrGetProcedureAddress"];
    addChart(s, pres.charts.BAR, [
      { name: "Training corpus", labels: calls, values: [8.9, 8.4, 0.1, 0.0, 0.1] },
      { name: "Unseen set", labels: calls, values: [0.8, 1.4, 6.8, 6.9, 20.8] },
    ], chartBase({
      x: 6.65, y: 1.55, w: 6.08, h: 3.75, barDir: "bar", barGrouping: "clustered", chartColors: [HEX.base, HEX.red],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 10, valAxisMinVal: 0, valAxisMaxVal: 25, valAxisHidden: true, valGridLine: { style: "none" },
      showLegend: true, legendPos: "b", legendFontSize: 11, barGapWidthPct: 40, catAxisLabelFontSize: 11, showTitle: true, title: "Share of API calls (%), Figure 5.9", titleFontSize: 12,
    }));
    card(s, MX, 5.35, 12.13, 1.4, HEX.redTint, "result band");
    [["110 / 110", "unseen threats detected"], ["0.865", "lowest threat probability"], ["6 / 200", "benign traces flagged"]].forEach(([n, l], i) => {
      const x = MX + 0.3 + i * 2.75;
      txt(s, n, { x, y: 5.45, w: 2.6, h: 0.6, fontSize: 26, bold: true, color: RED, fontFace: "Georgia", valign: "middle" });
      txt(s, l, { x, y: 6.07, w: 2.6, h: 0.5, fontSize: 12, color: INK });
    });
    txt(s, "The gate shifts toward structure (0.41) and timing (0.33); semantics keeps 0.25, still reading meaning from unfamiliar names.",
      { x: 8.85, y: 5.45, w: 3.7, h: 1.2, fontSize: 12.5, color: INK, valign: "middle" });
  }

  // ---------------- 12. RQ4 INTERPRETABILITY ----------------
  {
    const s = content("03 · Results · interpretability (RQ4)", "Every alert can be opened and read",
      "[~55 s] RQ4. An explanation has two levels, answering two analyst questions: which kind of evidence carried the verdict, and what that evidence was. Here is one held-out malware sample; the framework calls it a threat and gives its malware family 0.95. " +
      "Level one is the gate: it leaned on timing, 0.39, with structure and semantics close behind, so all three contributed. Level two is gradient-times-input saliency on the calls: handle duplication, atom tables, socket configuration, event synchronisation, hashing and file attributes. " +
      "They are spread across the trace, yet each sits in a burst, which is the temporal evidence behind the gate weight, and all are anchored to the sample's own process, the one relation the ablation found load-bearing. " +
      "Both levels are quantities the model computed while deciding, not an explainer fitted afterwards: faithful to the computation, not a proof of cause, and testable against the trace.", 2);
    answers(s, ["RQ4"]);
    addChart(s, pres.charts.BAR, [{ name: "Gate weight", labels: ["Temporal", "Structural", "Semantic"], values: [0.39, 0.31, 0.29] }],
      chartBase({
        x: MX, y: 1.55, w: 3.75, h: 3.95, barDir: "col", chartColors: [HEX.tmp, HEX.str, HEX.sem], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
        dataLabelFontSize: 13, dataLabelFontBold: true, valAxisMinVal: 0, valAxisMaxVal: 0.5, valAxisHidden: true, valGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 45,
        showTitle: true, title: "Level 1 · which dimension (gate)", titleFontSize: 12,
      }));
    const sal = [["NtOpenSection", 0.78], ["NtQueryInformationFile", 0.79], ["CryptDestroyHash", 0.79], ["CryptCreateHash", 0.80], ["NtSetInformationFile", 0.80],
                 ["NtOpenEvent", 0.82], ["setsockopt", 0.82], ["NtFindAtom", 0.82], ["NtAddAtomEx", 0.84], ["NtDuplicateObject", 1.00]];
    addChart(s, pres.charts.BAR, [{ name: "Relative saliency", labels: sal.map(x => x[0]), values: sal.map(x => x[1]) }],
      chartBase({
        x: 4.55, y: 1.55, w: 8.18, h: 3.95, barDir: "bar", chartColors: [HEX.grey], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 10,
        valAxisMinVal: 0, valAxisMaxVal: 1.15, valAxisHidden: true, valGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 35, catAxisLabelFontSize: 10,
        showTitle: true, title: "Level 2 · which calls (gradient × input saliency, Figure 5.11)", titleFontSize: 12,
      }));
    card(s, MX, 5.65, 5.95, 1.1, LIGHT, "example");
    txt(s, "Held-out malware, family probability 0.95. Cryptographic, networking, synchronisation and file capabilities, each in a burst, all anchored to the sample's own process.",
      { x: MX + 0.25, y: 5.65, w: 5.5, h: 1.1, fontSize: 12.5, valign: "middle" });
    card(s, 6.78, 5.65, 5.95, 1.1, HEX.redTint, "faithfulness");
    txt(s, "Both levels are quantities the model computed while deciding, not an explainer fitted afterwards: faithful to the computation, not a proof of cause.",
      { x: 7.03, y: 5.65, w: 5.5, h: 1.1, fontSize: 12.5, valign: "middle", color: INK });
  }

  // ---------------- 13. CONCLUSION: ANSWERS + LIMITATIONS + FUTURE WORK ----------------
  sec("Conclusion");
  {
    const s = content("04 · Conclusion", "Answers, limitations and future work",
      "[~90 s] To draw it together. RQ1: yes, the semantic branch reads the function of a call rather than its name, bounded to behaviour-preserving substitution. RQ2: yes, all three branches are used and any one can be lost, while only the full model names the class best; that holds for a missing branch, and the attacked case is RQ3. " +
      "RQ3: on clean traces it matches the baselines, which I state openly; under behaviour-preserving attacks it alone keeps detecting, missing at most two threats, while each baseline fails once the one dimension it depends on is disturbed. It is the only one of the five that continues to work. " +
      "RQ4: yes, at two levels, read from the model's own quantities. RQ5: yes, all 110 unseen-source threats, across one genuine shift of source. " +
      "The bounds: verdicts come after a complete trace; the BiLSTM's memory is finite on the longest traces; a two-source Windows snapshot with few ransomware samples; the benign side of the generalisation test used familiar software; evasive samples; the cost of per-call embeddings; representative attacks and one training run; and a gate weight is a coefficient, not a share. They qualify the breadth of the claims, not their validity. " +
      "Every next step extends the same architecture, first detection on the fly with a continuous-time dynamic graph. The contribution is one end-to-end model whose adaptive gate reweighs structure, meaning and timing for every decision.", 3);
    const a = [
      ["RQ1", "Yes", "Swaps barely move the intact model; with structure silenced, meaning-preserving swaps change 0.7 % of predictions, meaning-destroying 5.3 %. Bounded to behaviour-preserving substitution.", "Slide 9"],
      ["RQ2", "Yes", "All three branches used, leadership shifting by class; any one can be lost (≥ 95.8 % threat recall); only the full model reaches 285 / 300 exact class. Holds for a missing branch, not an attacked one.", "Slides 8–9"],
      ["RQ3", "Alone holds under attack", "On clean traces it matches the baselines. Under behaviour-preserving attacks it alone keeps detecting (at most 2 threats missed): each baseline fails once the one dimension it depends on is disturbed.", "Slide 10"],
      ["RQ4", "Yes", "Two levels: the gate names the deciding dimension; saliency and the graph name the calls, when each ran and what it operated on. Faithful to the computation, not a proof of cause.", "Slides 8, 12"],
      ["RQ5", "Yes", "110 / 110 unseen-source threats detected, none near the boundary, though 27.8 % of their distinct calls never appear in training. One shift of source, not breadth across families.", "Slide 11"],
    ];
    a.forEach(([tag, v, t, ev], i) => {
      const y = 1.5 + i * 0.64;
      card(s, MX, y, 12.13, 0.57, LIGHT, tag + " answer");
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.12, y: y + 0.11, w: 0.72, h: 0.35, fill: { color: RED }, line: { color: RED, width: 0 }, rectRadius: 0.06, objectName: tag + " tag" });
      txt(s, tag, { x: MX + 0.12, y: y + 0.11, w: 0.72, h: 0.35, fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle" });
      txt(s, v, { x: MX + 0.98, y, w: 1.3, h: 0.57, fontSize: 13, bold: true, color: RED, valign: "middle", fontFace: "Georgia" });
      txt(s, t, { x: MX + 2.3, y, w: 8.75, h: 0.57, fontSize: 11.5, valign: "middle" });
      txt(s, ev, { x: MX + 11.05, y, w: 0.95, h: 0.57, fontSize: 10, bold: true, color: GREY, valign: "middle", align: "right" });
    });
    // limitations (§6.2) and future work (§6.3)
    const L = [
      ["Offline: ", "verdict after a complete trace"],
      ["Memory: ", "BiLSTM horizon on the longest traces"],
      ["Snapshot: ", "two sources, Windows, few ransomware"],
      ["Provenance: ", "RQ5 benign side was familiar software"],
      ["Evasion: ", "sandbox-aware malware hides behaviour"],
      ["Cost: ", "a 768-d embedding on every call"],
      ["Evaluation: ", "no adaptive adversary; a single run"],
      ["Attribution: ", "gate weight ≠ share of the decision"],
    ];
    card(s, MX, 4.78, 7.45, 1.97, LIGHT, "limitations");
    txt(s, "Limitations (§6.2)", { x: MX + 0.2, y: 4.84, w: 4, h: 0.3, fontSize: 13, bold: true, color: RED, fontFace: "Georgia" });
    [L.slice(0, 4), L.slice(4)].forEach((col, k) => {
      txt(s, col.flatMap(([h, b], j) => [{ text: h, options: { bold: true } }, { text: b, options: { breakLine: j < col.length - 1 } }]),
        { x: MX + 0.2 + k * 3.6, y: 5.2, w: 3.55, h: 1.45, fontSize: 10.5, paraSpaceAfter: 6 });
    });
    const F = [
      ["Detection on the fly: ", "continuous-time dynamic graph"],
      ["Temporal backbone: ", "transformer and Mamba vs BiLSTM"],
      ["Breadth: ", "more sandboxes and sources; calibration"],
      ["Stronger evaluation: ", "adaptive adversary; repeated seeds"],
      ["Semantic ceiling: ", "a zero-shot LLM baseline"],
    ];
    card(s, 8.2, 4.78, 4.53, 1.97, HEX.redTint, "future work");
    txt(s, "Future work (§6.3)", { x: 8.4, y: 4.84, w: 4, h: 0.3, fontSize: 13, bold: true, color: RED, fontFace: "Georgia" });
    txt(s, F.flatMap(([h, b], j) => [{ text: h, options: { bold: true } }, { text: b, options: { breakLine: j < F.length - 1 } }]),
      { x: 8.4, y: 5.2, w: 4.25, h: 1.45, fontSize: 10.5, paraSpaceAfter: 3 });
  }

  // ---------------- 14. CLOSING ----------------
  {
    const s = pres.addSlide({ masterName: "CLOSING", sectionTitle: section });
    s.addImage({ path: A + "logo_white.png", x: MX + 0.2, y: 0.55, w: 2.3, h: 0.70, altText: "Universität Koblenz logo" });
    txt(s, "Thank you", { x: MX + 0.2, y: 2.5, w: 9, h: 1.3, fontSize: 60, bold: true, color: WHITE, fontFace: "Georgia", valign: "middle" });
    txt(s, "Questions and discussion", { x: MX + 0.2, y: 3.85, w: 9, h: 0.6, fontSize: 24, color: WHITE });
    txt(s, "Kandenghat Abhishek Menon  ·  Universität Koblenz", { x: MX + 0.2, y: 6.3, w: 9, h: 0.4, fontSize: 14, color: WHITE });
    const N = [[10.4, 2.1, 0.4], [11.9, 2.9, 0.26], [10.9, 3.9, 0.55], [12.3, 4.6, 0.3], [9.8, 4.9, 0.28], [11.5, 5.6, 0.22]];
    const E = [[0, 2], [1, 2], [2, 3], [2, 4], [3, 5], [4, 5], [0, 1]];
    for (const [a, b] of E) {
      const [x1, y1] = N[a], [x2, y2] = N[b];
      const o = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001, line: { color: WHITE, width: 1.25, transparency: 55 }, objectName: "motif edge" };
      if ((x2 < x1) !== (y2 < y1)) o.flipH = true;
      s.addShape(pres.shapes.LINE, o);
    }
    N.forEach(([x, y, d], i) => s.addShape(pres.shapes.OVAL, { x: x - d / 2, y: y - d / 2, w: d, h: d, fill: { color: WHITE, transparency: i === 2 ? 0 : 35 }, line: { color: WHITE, width: 0 }, objectName: "motif node" }));
    s.addNotes("[~10 s] Thank you. I am happy to take questions. (Then switch to the live demo.)");
  }

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch(e => { console.error(e); process.exit(1); });

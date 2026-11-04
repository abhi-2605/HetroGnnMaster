// Build the presenting deck: one full-bleed picture per PDF page, speaker notes copied over.
// Usage: node make_image_pptx.js <dir with sNN.png + notes.json> <output.pptx>
const pptxgen = require("pptxgenjs");
const dir = process.argv[2], out = process.argv[3];
const notes = require(require("path").resolve(dir, "notes.json"));
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Threat Detection Using Heterogeneous Graph Neural Networks with Semantic and Temporal Features";
pres.author = "Kandenghat Abhishek Menon";
notes.forEach((n, i) => {
  const s = pres.addSlide();
  s.addImage({ path: `${dir}/s${String(i + 1).padStart(2, "0")}.png`, x: 0, y: 0, w: 13.333, h: 7.5, altText: "Slide " + (i + 1) });
  if (n) s.addNotes(n);
});
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out));

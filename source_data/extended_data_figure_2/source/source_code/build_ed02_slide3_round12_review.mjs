import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "file:///D:/combineddata/github_clean_repo/tmp/ed02_round7_typography_2026-08-08/deck/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const ROOT = "D:/combineddata/github_clean_repo";
const RECORD = path.join(ROOT, "docs/figure_version_records/extended_data_fig02_2026-08-09_round12_vector_vor");
const PANEL_ROOT = path.join(ROOT, "tmp/ed02_round12_fix_2026-08-09/panels");
const VECTOR_ROOT = path.join(PANEL_ROOT, "ppt_vector");
const OUT = path.join(ROOT, "tmp/ed02_round12_fix_2026-08-09/deck");
const INPUTS = [
  {
    tag: "cumulative",
    input: path.join(RECORD, "review_evidence/sup-figs-v2_ed02-slide3_round11_review.pptx"),
    output: path.join(OUT, "sup-figs-v2_ed02-slide3_round12_review.pptx"),
    slideIndex: 2,
    expectedSlides: 12,
  },
  {
    tag: "review",
    input: path.join(RECORD, "review_evidence/ed02-slide3_round11_review.pptx"),
    output: path.join(OUT, "ed02-slide3_round12_review.pptx"),
    slideIndex: 0,
    expectedSlides: 1,
  },
];
const CHANGES = [
  {
    key: "a",
    objectName: "ED2 panel a: four branch-policy definitions",
    stem: "ed02a_four_policy_schematic_round11",
    alt: "Panel a: four branch-policy definitions; grid, arrows, arrowheads, and text scaled proportionally; preceding-turn annotation clears the grid",
  },
  {
    key: "g",
    objectName: "ED2 panel g: depth-dependent P_BS profile and full-day contrasts",
    stem: "ed02g_depth_pbs_profile_round11",
    alt: "Panel g: depth-dependent P_BS profile and full-day contrasts; compact shallow, middle, and deep positions; y-axis titles clear the tick labels",
  },
];

async function sha256(filePath) {
  return crypto.createHash("sha256").update(await fs.readFile(filePath)).digest("hex");
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function parseNdjson(text) {
  return text.split(/\r?\n/).filter((line) => line.trim()).map((line) => JSON.parse(line));
}

async function editDeck(spec) {
  const presentation = await PresentationFile.importPptx(await FileBlob.load(spec.input));
  if (presentation.slides.items.length !== spec.expectedSlides) {
    throw new Error(`${spec.tag}: expected ${spec.expectedSlides} slides`);
  }
  const slide = presentation.slides.items[spec.slideIndex];
  const before = await presentation.inspect({ kind: "slide,image,textbox,shape,notes,layout", maxChars: 500000 });
  const records = parseNdjson(before.ndjson);
  const displaySlide = spec.slideIndex + 1;
  const images = records.filter((record) => record.kind === "image" && record.slide === displaySlide);
  if (images.length !== 10) throw new Error(`${spec.tag}: expected ten vector image objects`);

  const replaced = [];
  for (const change of CHANGES) {
    const imageRecord = images.find((record) => record.name === change.objectName);
    if (!imageRecord) throw new Error(`${spec.tag}: missing ${change.objectName}`);
    const vectorPath = path.join(VECTOR_ROOT, `${change.stem}.svg`);
    const target = presentation.resolve(imageRecord.id);
    const frame = { ...target.frame };
    target.replace({
      blob: await fs.readFile(vectorPath),
      contentType: "image/svg+xml",
      alt: change.alt,
      fit: "contain",
    });
    target.name = change.objectName;
    target.frame = frame;
    target.crop = { left: 0, top: 0, right: 0, bottom: 0 };
    target.fit = "contain";
    target.rotation = 0;
    target.flipHorizontal = false;
    target.flipVertical = false;
    target.lockAspectRatio = true;
    replaced.push({ key: change.key, frame, sha256: await sha256(vectorPath) });
  }

  slide.speakerNotes.textFrame.setText([
    "Extended Data Figure 2. Round-12 source-level display correction; exactly panels a-i and no panel j.",
    "Panel a retains the accepted 0.8x schematic geometry and now also scales arrow strokes, arrowheads, and all in-schematic text by 0.8; the preceding-turn annotation clears the grid.",
    "Panel g preserves the accepted fixed-30-minute values and axis limits; shallow, middle, and deep positions are compacted and both P_BS y-axis titles clear their tick labels.",
    "All other panels, the shared b/c legend, the exact panel frames, the panel-letter frames, the panel inventory, and all 16 plot-ready tables are unchanged.",
    "Scope: first 30 cumulative TimeInMaze minutes; all modes advance the clock; junction decisions are Explore mode only. Behavior-only; manual neural masks are not applicable.",
    "Vector-first delivery: source-derived SVG objects in PowerPoint, editable SVG/PDF masters, and 600-dpi PNG/TIFF companions.",
    "[Sources]",
    "- Current input record: docs/figure_version_records/extended_data_fig02_2026-08-09_round12_vector_vor/",
    "- Corrected renderer: scripts/render_ed02_slide3_schematic_g_fix_round12.py",
  ]);
  slide.speakerNotes.setVisible(true);

  const after = await presentation.inspect({ kind: "slide,image,textbox,shape,notes,layout", maxChars: 500000 });
  const afterRecords = parseNdjson(after.ndjson);
  const afterImages = afterRecords.filter((record) => record.kind === "image" && record.slide === displaySlide);
  if (afterImages.length !== 10) throw new Error(`${spec.tag}: post-edit vector inventory changed`);
  await fs.writeFile(path.join(OUT, `${spec.tag}-after-inspect.ndjson`), `${after.ndjson}\n`, "utf8");
  await writeBlob(path.join(OUT, `${spec.tag}-artifact-preview.png`), await presentation.export({ slide, format: "png", scale: 2 }));
  await fs.writeFile(path.join(OUT, `${spec.tag}.layout.json`), await (await slide.export({ format: "layout" })).text(), "utf8");
  await PresentationFile.exportPptx(presentation).then((blob) => blob.save(spec.output));
  return {
    tag: spec.tag,
    input: path.relative(ROOT, spec.input).replaceAll("\\", "/"),
    output: path.basename(spec.output),
    outputSha256: await sha256(spec.output),
    slideCount: presentation.slides.items.length,
    editedSlide: displaySlide,
    vectorImageCount: afterImages.length,
    replaced,
  };
}

async function main() {
  await fs.mkdir(OUT, { recursive: true });
  const results = [];
  for (const input of INPUTS) results.push(await editDeck(input));
  const report = {
    status: "pass",
    reviewRound: 12,
    authoringTool: "@oai/artifact-tool",
    scope: "replace panels a and g only; retain every frame and all other slide objects",
    results,
  };
  await fs.writeFile(path.join(OUT, "round12-build-report.json"), `${JSON.stringify(report, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => { console.error(error); process.exitCode = 1; });

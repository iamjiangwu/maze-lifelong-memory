import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "file:///D:/combineddata/github_clean_repo/tmp/ed02_round7_typography_2026-08-08/deck/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const ROOT = "D:/combineddata/github_clean_repo";
const RECORD = path.join(ROOT, "docs/figure_version_records/extended_data_fig02_2026-08-09_round14_schematic_g_fix_vor");
const PANEL_ROOT = path.join(ROOT, "tmp/ed02_round13_g_labels_2026-08-09/panels");
const VECTOR_ROOT = path.join(PANEL_ROOT, "ppt_vector");
const OUT = path.join(ROOT, "tmp/ed02_round13_g_labels_2026-08-09/deck");
const PANEL_G_NAME = "ED2 panel g: depth-dependent P_BS profile and full-day contrasts";
const PANEL_G_LABEL_NAME = "ED2 panel label g";
const PANEL_G_VECTOR = path.join(VECTOR_ROOT, "ed02g_depth_pbs_profile_round11.svg");
const PANEL_G_LABEL_TOP = 463.0;
const INPUTS = [
  {
    tag: "cumulative",
    input: path.join(RECORD, "review_evidence/sup-figs-v2_ed02-slide3_round12_review.pptx"),
    output: path.join(OUT, "sup-figs-v2_ed02-slide3_round13_review.pptx"),
    slideIndex: 2,
    expectedSlides: 12,
  },
  {
    tag: "review",
    input: path.join(RECORD, "review_evidence/ed02-slide3_round12_review.pptx"),
    output: path.join(OUT, "ed02-slide3_round13_review.pptx"),
    slideIndex: 0,
    expectedSlides: 1,
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
  const displaySlide = spec.slideIndex + 1;
  const before = await presentation.inspect({ kind: "slide,image,textbox,shape,notes,layout", maxChars: 500000 });
  const beforeRecords = parseNdjson(before.ndjson);
  const images = beforeRecords.filter((record) => record.kind === "image" && record.slide === displaySlide);
  if (images.length !== 10) throw new Error(`${spec.tag}: expected ten vector image objects`);
  const imageRecord = images.find((record) => record.name === PANEL_G_NAME);
  const labelRecord = beforeRecords.find(
    (record) => record.kind === "textbox" && record.slide === displaySlide && record.name === PANEL_G_LABEL_NAME,
  );
  if (!imageRecord || !labelRecord) throw new Error(`${spec.tag}: missing panel-g image or label`);

  const image = presentation.resolve(imageRecord.id);
  const originalImageFrame = { ...image.frame };
  image.replace({
    blob: await fs.readFile(PANEL_G_VECTOR),
    contentType: "image/svg+xml",
    alt: "Panel g: depth-dependent P_BS profile and full-day contrasts; maintained Shallow 1-2, Middle 3-4, and Deep 5 labels restored; values and frame unchanged",
    fit: "contain",
  });
  image.name = PANEL_G_NAME;
  image.frame = originalImageFrame;
  image.crop = { left: 0, top: 0, right: 0, bottom: 0 };
  image.fit = "contain";
  image.rotation = 0;
  image.flipHorizontal = false;
  image.flipVertical = false;
  image.lockAspectRatio = true;

  const label = presentation.resolve(labelRecord.id);
  const originalLabelFrame = { ...label.frame };
  label.frame = { ...originalLabelFrame, top: PANEL_G_LABEL_TOP };
  label.name = PANEL_G_LABEL_NAME;

  slide.speakerNotes.textFrame.setText([
    "Extended Data Figure 2. Round-13 source-level panel-g label correction; exactly panels a-i and no panel j.",
    "Panel g restores the maintained categorical x labels Shallow 1-2, Middle 3-4, and Deep 5 after a shared-x display transform erased them.",
    "Only the external panel letter g moves upward by one 10.5-pt character height, from slide y=473.5 to y=463.0. The panel-g values, axes, frame, crop, colors, markers, and every other panel object are unchanged.",
    "Scope: first 30 cumulative TimeInMaze minutes; all modes advance the clock; junction decisions are Explore mode only. Behavior-only; manual neural masks are not applicable.",
    "Vector-first delivery: source-derived SVG objects in PowerPoint, editable SVG/PDF masters, and 600-dpi PNG/TIFF companions.",
    "[Sources]",
    "- Input record: docs/figure_version_records/extended_data_fig02_2026-08-09_round14_schematic_g_fix_vor/",
    "- Superseding record: docs/figure_version_records/extended_data_fig02_2026-08-09_round15_panel_g_labels_vor/",
    "- Corrected renderer: scripts/render_ed02_slide3_panel_g_labels_round13.py",
  ]);
  slide.speakerNotes.setVisible(true);

  const after = await presentation.inspect({ kind: "slide,image,textbox,shape,notes,layout", maxChars: 500000 });
  const afterRecords = parseNdjson(after.ndjson);
  const afterImages = afterRecords.filter((record) => record.kind === "image" && record.slide === displaySlide);
  const afterLabel = afterRecords.find(
    (record) => record.kind === "textbox" && record.slide === displaySlide && record.name === PANEL_G_LABEL_NAME,
  );
  if (afterImages.length !== 10 || !afterLabel) throw new Error(`${spec.tag}: post-edit inventory changed`);
  if (Math.abs(afterLabel.bbox[1] - PANEL_G_LABEL_TOP) > 0.01) {
    throw new Error(`${spec.tag}: panel-g label top mismatch ${afterLabel.bbox[1]}`);
  }

  await fs.mkdir(OUT, { recursive: true });
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
    panelG: {
      frame: originalImageFrame,
      svgSha256: await sha256(PANEL_G_VECTOR),
      depthLabels: ["Shallow 1-2", "Middle 3-4", "Deep 5"],
    },
    panelGLabel: {
      previousFrame: originalLabelFrame,
      newFrame: { ...originalLabelFrame, top: PANEL_G_LABEL_TOP },
      deltaTop: PANEL_G_LABEL_TOP - originalLabelFrame.top,
    },
  };
}

async function main() {
  await fs.mkdir(OUT, { recursive: true });
  const results = [];
  for (const input of INPUTS) results.push(await editDeck(input));
  const report = {
    status: "pass",
    reviewRound: 13,
    authoringTool: "@oai/artifact-tool",
    scope: "replace panel g and move only panel label g by -10.5; retain every scientific frame and all other slide objects",
    results,
  };
  await fs.writeFile(path.join(OUT, "round13-build-report.json"), `${JSON.stringify(report, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => { console.error(error); process.exitCode = 1; });

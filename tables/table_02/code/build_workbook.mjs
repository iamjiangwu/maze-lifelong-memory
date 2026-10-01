import crypto from "node:crypto";
import fs from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dependencyRoot = process.env.ARTIFACT_TOOL_NODE_MODULES;
if (!dependencyRoot) {
  throw new Error("Set ARTIFACT_TOOL_NODE_MODULES to the dependency runtime's node_modules directory.");
}
const require = createRequire(path.join(dependencyRoot, "__artifact_tool_loader__.cjs"));
const { SpreadsheetFile, Workbook } = require("@oai/artifact-tool");

const here = path.dirname(fileURLToPath(import.meta.url));
const recordRoot = path.resolve(here, "..");
const outputRoot = path.resolve(process.argv[2] ?? path.join(recordRoot, "build"));
const sourceRoot = path.resolve(process.argv[3] ?? path.join(recordRoot, "source_data"));
await fs.mkdir(outputRoot, { recursive: true });
const sourcePath = path.join(sourceRoot, "comprehensive_interval_ledger.csv");
const sourceManifestPath = path.join(sourceRoot, "comprehensive_interval_ledger_manifest.json");
const workbookPath = path.join(outputRoot, "Supplementary_Table_2_Long_Gap_Intervals.xlsx");
const companionCsvPath = path.join(outputRoot, "Supplementary_Table_2_Long_Gap_Intervals.csv");
const manifestPath = path.join(outputRoot, "MANIFEST.json");
const validationPath = path.join(outputRoot, "VALIDATION.md");
const sentencePath = path.join(outputRoot, "manuscript_sentence.tex");
const inspectionPath = `${workbookPath}.inspect.ndjson`;

const excludedPair = "1616day1->1616bday1";
const excluded180Pair = "618-6day1->618-12day1";
const retained1616Pair = "1616bday1->1616b-3day1";
const manuscriptSentence =
  "Day~$i$ of the first cycle was the animal's naive session; subsequent Day~$i$ sessions were generally return sessions conducted after approximately 90~d without maze experience. Across 56 included long-gap intervals, the mean $\\pm$ s.d. was 90 $\\pm$ 4~d (range, 82--110~d). Mouse 1616's 14-d interval and mouse 618's 180-d interval were excluded from this calculation; actual interval lengths are summarized in Supplementary Table~2.";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function parseCsv(text) {
  const records = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
    } else if (character === '"') {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n") {
      row.push(field.replace(/\r$/, ""));
      records.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    records.push(row);
  }
  const populated = records.filter((values) => values.some((value) => value !== ""));
  const [headers, ...data] = populated;
  return data.map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])),
  );
}

function sha256(text) {
  return crypto.createHash("sha256").update(text).digest("hex");
}

function parseBoolean(value) {
  return String(value).toLowerCase() === "true";
}

function parseDate(value) {
  assert(/^\d{4}-\d{2}-\d{2}$/.test(value), `Expected calendar date: ${value}`);
  const normalized = `${value}T00:00:00Z`;
  const parsed = new Date(normalized);
  assert(!Number.isNaN(parsed.valueOf()), `Invalid date: ${value}`);
  return parsed;
}

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function sampleSd(values) {
  const center = mean(values);
  return Math.sqrt(
    values.reduce((total, value) => total + (value - center) ** 2, 0) /
      (values.length - 1),
  );
}

const sourceText = await fs.readFile(sourcePath, "utf8");
const sourceRows = parseCsv(sourceText);
const sourceManifestText = await fs.readFile(sourceManifestPath, "utf8");
const sourceManifest = JSON.parse(sourceManifestText);

assert(sourceRows.length === 58, `Expected 58 displayed intervals, found ${sourceRows.length}.`);
assert(new Set(sourceRows.map((row) => row["Animal ID"])).size === 18, "Expected 18 mice.");
assert(
  sourceRows.filter((row) => parseBoolean(row["Fixed-month behavior return roster"])).length === 56,
  "Expected 56 behavior intervals.",
);
assert(
  sourceRows.filter((row) => parseBoolean(row["Maintained calcium imaging"])).length === 37,
  "Expected 37 calcium-imaged intervals (36 long-gap and one short).",
);
assert(sourceRows.filter((row) => row["Pair key"] === excludedPair).length === 1, "Expected one short 1616 pair.");
assert(
  sourceRows.filter((row) => row["Pair key"] === retained1616Pair && row["Return month"] === "6").length === 1,
  "Expected the retained 1616 interval in the 6-month column.",
);
assert(
  sourceRows.filter((row) => row["Pair key"] === excluded180Pair && row["Return month"] === "12").length === 1,
  "Expected the 618 180-d interval to remain in the 12-month column.",
);

const animals = [...new Set(sourceRows.map((row) => row["Animal ID"]))].sort(
  (left, right) => Number(left) - Number(right),
);
const behaviorRows = sourceRows.filter((row) => parseBoolean(row["Fixed-month behavior return roster"]));
const overlapRows = behaviorRows.filter((row) => parseBoolean(row["Maintained calcium imaging"]));
const approximatelyThreeMonthRows = sourceRows.filter(
  (row) => ![excludedPair, excluded180Pair].includes(row["Pair key"]),
);
const approximatelyThreeMonthValues = approximatelyThreeMonthRows.map((row) =>
  Number(row["Interval (d)"]),
);

assert(animals.length === 18, "Publication matrix animal count mismatch.");
assert(overlapRows.length === 35, "Behavior-calcium overlap count mismatch.");
assert(approximatelyThreeMonthRows.length === 56, "Approximately 3-month scope mismatch.");
assert(
  Math.abs(mean(approximatelyThreeMonthValues) - 90.08928571428571) < 1e-10,
  "Approximately 3-month mean mismatch.",
);
assert(
  Math.abs(sampleSd(approximatelyThreeMonthValues) - 3.56384259276052) < 1e-10,
  "Approximately 3-month sample s.d. mismatch.",
);

const matrixRows = animals.map((animal) => {
  const rows = sourceRows.filter((row) => row["Animal ID"] === animal);
  const additionalCalcium = rows.some(
    (row) =>
      parseBoolean(row["Maintained calcium imaging"]) &&
      !["3", "6", "9", "12"].includes(row["Return month"]),
  );
  const calciumMonths = rows
    .filter(
      (row) =>
        parseBoolean(row["Maintained calcium imaging"]) &&
        ["3", "6", "9", "12"].includes(row["Return month"]),
    )
    .sort((left, right) => Number(left["Return month"]) - Number(right["Return month"]))
    .map((row) => row["Return month"]);
  const calciumLabel = calciumMonths.length
    ? `${calciumMonths.join(", ")}${additionalCalcium ? "; other interval" : ""}`
    : additionalCalcium
      ? "Other intervals"
      : "None";
  return { animal, calciumLabel };
});

const workbook = Workbook.create();
workbook.comments.setSelf({ displayName: "User" });
const publicationSheet = workbook.worksheets.add("Supplementary Table 2");
const sourceSheet = workbook.worksheets.add("Source Data");
const summarySheet = workbook.worksheets.add("Summary and provenance");

const ink = "#202020";
const navy = "#1F4168";
const headerFill = "#E4E4E4";
const animalFill = "#F3F3F3";
const paleFill = "#F7F8FA";
const border = "#C8C8C8";
const muted = "#555E68";
const warningFill = "#FFF2CC";

// Sheet 1: the reader-facing table corresponding to the final PDF.
publicationSheet.showGridLines = false;
publicationSheet.getRange("A1:G1").merge();
publicationSheet.getRange("A1").values = [[
  "Supplementary Table 2. Intervals between successive recording cycles",
]];
publicationSheet.getRange("A1:G1").format = {
  font: { bold: true, color: ink, size: 17 },
  verticalAlignment: "center",
};
publicationSheet.getRange("A1:G1").format.rowHeight = 34;

const publicationHeaders = [
  "Mouse",
  "3-month return\n(d)",
  "6-month return\n(d)",
  "9-month return\n(d)",
  "12-month return\n(d)",
  "Other intervals\n(d)",
  "Calcium-imaged returns\n(months)",
];
publicationSheet.getRange("A3:G3").values = [publicationHeaders];
publicationSheet.getRange("A3:G3").format = {
  fill: headerFill,
  font: { bold: true, color: ink, size: 10 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    top: { style: "thin", color: "#555555" },
    bottom: { style: "thin", color: "#555555" },
    insideVertical: { style: "thin", color: border },
  },
};
publicationSheet.getRange("A3:G3").format.rowHeight = 42;

const firstPublicationRow = 4;
const lastPublicationRow = firstPublicationRow + matrixRows.length - 1;
const shortSourceExcelRow = sourceRows.findIndex((row) => row["Pair key"] === excludedPair) + 2;
assert(shortSourceExcelRow >= 2, "Short mouse-1616 source row is missing.");
publicationSheet.getRange(`A${firstPublicationRow}:A${lastPublicationRow}`).values = matrixRows.map(
  (row) => [row.animal],
);
for (let index = 0; index < matrixRows.length; index += 1) {
  const excelRow = firstPublicationRow + index;
  for (const [column, month] of [
    ["B", 3],
    ["C", 6],
    ["D", 9],
    ["E", 12],
  ]) {
    publicationSheet.getRange(`${column}${excelRow}`).formulas = [[
      `=IFERROR(SUMIFS('Source Data'!$I$2:$I$59,'Source Data'!$A$2:$A$59,$A${excelRow},'Source Data'!$B$2:$B$59,${month})/COUNTIFS('Source Data'!$A$2:$A$59,$A${excelRow},'Source Data'!$B$2:$B$59,${month}),"")`,
    ]];
  }
  publicationSheet.getRange(`F${excelRow}`).formulas = [[
    matrixRows[index].animal === "1616"
      ? `='Source Data'!I${shortSourceExcelRow}`
      : `=IFERROR(SUMIFS('Source Data'!$I$2:$I$59,'Source Data'!$A$2:$A$59,$A${excelRow},'Source Data'!$J$2:$J$59,"No")/COUNTIFS('Source Data'!$A$2:$A$59,$A${excelRow},'Source Data'!$J$2:$J$59,"No"),"")`,
  ]];
  publicationSheet.getRange(`G${excelRow}`).values = [[matrixRows[index].calciumLabel]];
}
publicationSheet.getRange(`B${firstPublicationRow}:F${lastPublicationRow}`).format.numberFormat = "0";
publicationSheet.getRange(`A${firstPublicationRow}:G${lastPublicationRow}`).format = {
  font: { color: ink, size: 10 },
  verticalAlignment: "center",
  borders: {
    insideHorizontal: { style: "thin", color: border },
    insideVertical: { style: "thin", color: border },
    bottom: { style: "thin", color: "#555555" },
  },
};
publicationSheet.getRange(`A${firstPublicationRow}:A${lastPublicationRow}`).format = {
  fill: animalFill,
  font: { bold: true, color: ink, size: 10 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
publicationSheet.getRange(`B${firstPublicationRow}:F${lastPublicationRow}`).format.horizontalAlignment =
  "center";
publicationSheet.getRange(`G${firstPublicationRow}:G${lastPublicationRow}`).format.horizontalAlignment =
  "left";
publicationSheet.getRange(`A${firstPublicationRow}:G${lastPublicationRow}`).format.rowHeight = 22;

const footnotes = JSON.parse(await fs.readFile(path.join(recordRoot,'source_data','publication_notes.json'),'utf8'));
for (let index = 0; index < footnotes.length; index += 1) {
  const excelRow = lastPublicationRow + 2 + index;
  publicationSheet.getRange(`A${excelRow}:G${excelRow}`).merge();
  publicationSheet.getRange(`A${excelRow}`).values = [[footnotes[index]]];
  publicationSheet.getRange(`A${excelRow}:G${excelRow}`).format = {
    fill: paleFill,
    font: { italic: false, color: ink, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  publicationSheet.getRange(`A${excelRow}:G${excelRow}`).format.rowHeight = index === 3 ? 58 : 42;
}
publicationSheet.getRange("A1:A27").format.columnWidth = 11;
publicationSheet.getRange("B1:E27").format.columnWidth = 16;
publicationSheet.getRange("F1:F27").format.columnWidth = 19;
publicationSheet.getRange("G1:G27").format.columnWidth = 28;
publicationSheet.freezePanes.freezeRows(3);

// Sheet 2: exact session-level source ledger.
sourceSheet.showGridLines = false;
const sourceHeaders = [
  "Animal ID",
  "Table return month (mo)",
  "Previous nominal month (mo)",
  "Preceding Day i session",
  "Preceding Day i+1 session",
  "Preceding Day i+1 date",
  "Return Day i session",
  "Return Day i date",
  "Interval (d)",
  "Fixed-month behavior return roster",
  "Maintained calcium imaging",
  "Pair key",
  "Schedule note",
  "Included in approximately 3-month summary",
];
sourceSheet.getRange("A1:N1").values = [sourceHeaders];
sourceSheet.getRange("A2:N59").values = sourceRows.map((row) => [
  row["Animal ID"],
  ["3", "6", "9", "12"].includes(row["Return month"])
    ? Number(row["Return month"])
    : null,
  Number.isFinite(Number(row["Previous nominal month"])) ? Number(row["Previous nominal month"]) : null,
  row["Preceding Day i session"],
  row["Preceding Day i+1 session"],
  parseDate(row["Preceding Day i+1 date"]),
  row["Return Day i session"],
  parseDate(row["Return Day i date"]),
  Number(row["Interval (d)"]),
  parseBoolean(row["Fixed-month behavior return roster"]) ? "Yes" : "No",
  parseBoolean(row["Maintained calcium imaging"]) ? "Yes" : "No",
  row["Pair key"],
  row["Schedule note"],
  [excludedPair, excluded180Pair].includes(row["Pair key"]) ? "No" : "Yes",
]);
sourceSheet.getRange("I2:I59").formulas = sourceRows.map((_, index) => [`=H${index + 2}-F${index + 2}`]);
sourceSheet.getRange("A1:N1").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF", size: 9 },
  wrapText: true,
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
sourceSheet.getRange("A1:N1").format.rowHeight = 42;
sourceSheet.getRange("F2:F59").format.numberFormat = "yyyy-mm-dd";
sourceSheet.getRange("H2:H59").format.numberFormat = "yyyy-mm-dd";
sourceSheet.getRange("I2:I59").format.numberFormat = "0";
sourceSheet.getRange("A2:N59").format = {
  font: { color: ink, size: 9 },
  borders: { insideHorizontal: { style: "thin", color: border } },
  verticalAlignment: "center",
};
sourceSheet.getRange("A2:C59").format.horizontalAlignment = "center";
sourceSheet.getRange("F2:F59").format.horizontalAlignment = "center";
sourceSheet.getRange("H2:K59").format.horizontalAlignment = "center";
sourceSheet.getRange("N2:N59").format.horizontalAlignment = "center";
sourceSheet.getRange("M2:N59").format.wrapText = true;
for (let index = 0; index < sourceRows.length; index += 1) {
  if (sourceRows[index]["Schedule note"]) {
    sourceSheet.getRange(`A${index + 2}:N${index + 2}`).format.rowHeight = 38;
  }
}
sourceSheet.getRange("A1:A59").format.columnWidth = 11;
sourceSheet.getRange("B1:C59").format.columnWidth = 16;
sourceSheet.getRange("D1:E59").format.columnWidth = 21;
sourceSheet.getRange("F1:F59").format.columnWidth = 21;
sourceSheet.getRange("G1:G59").format.columnWidth = 21;
sourceSheet.getRange("H1:H59").format.columnWidth = 21;
sourceSheet.getRange("I1:I59").format.columnWidth = 13;
sourceSheet.getRange("J1:K59").format.columnWidth = 18;
sourceSheet.getRange("L1:L59").format.columnWidth = 30;
sourceSheet.getRange("M1:M59").format.columnWidth = 48;
sourceSheet.getRange("N1:N59").format.columnWidth = 24;
sourceSheet.freezePanes.freezeRows(1);
const sourceTable = sourceSheet.tables.add("A1:N59", true, "ComprehensiveIntervalLedger");
sourceTable.style = "TableStyleMedium2";
sourceTable.showBandedRows = true;
sourceTable.showFilterButton = true;

// Sheet 3: compact, formula-driven validation and provenance.
summarySheet.showGridLines = false;
summarySheet.getRange("A1:G1").merge();
summarySheet.getRange("A1").values = [["Summary and provenance"]];
summarySheet.getRange("A1:G1").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF", size: 15 },
  verticalAlignment: "center",
};
summarySheet.getRange("A1:G1").format.rowHeight = 30;
summarySheet.getRange("A3:B3").values = [["Cohort and interval check", "Value"]];
summarySheet.getRange("A4:A17").values = [
  ["Displayed intervals"],
  ["Displayed mice"],
  ["Fixed-month behavior intervals"],
  ["Fixed-month behavior mice"],
  ["Calcium intervals (36 long + 1 short)"],
  ["Behavior and calcium intervals"],
  ["Behavior-only intervals"],
  ["Additional calcium interval(s)"],
  ["Approximately 3-month intervals"],
  ["Mean (d)"],
  ["Sample s.d. (d)"],
  ["Minimum (d)"],
  ["Maximum (d)"],
  ["Mouse 618 double-gap exception (d)"],
];
summarySheet.getRange("B4:B17").formulas = [
  ["=COUNT('Source Data'!$I$2:$I$59)"],
  ["=COUNTA(UNIQUE('Source Data'!$A$2:$A$59))"],
  ["=COUNTIF('Source Data'!$J$2:$J$59,\"Yes\")"],
  ["=COUNTA(UNIQUE(FILTER('Source Data'!$A$2:$A$59,'Source Data'!$J$2:$J$59=\"Yes\")))"],
  ["=COUNTIF('Source Data'!$K$2:$K$59,\"Yes\")"],
  ["=COUNTIFS('Source Data'!$J$2:$J$59,\"Yes\",'Source Data'!$K$2:$K$59,\"Yes\")"],
  ["=COUNTIFS('Source Data'!$J$2:$J$59,\"Yes\",'Source Data'!$K$2:$K$59,\"No\")"],
  ["=COUNTIFS('Source Data'!$J$2:$J$59,\"No\",'Source Data'!$K$2:$K$59,\"Yes\")"],
  ["=COUNTIF('Source Data'!$N$2:$N$59,\"Yes\")"],
  ["=AVERAGEIF('Source Data'!$N$2:$N$59,\"Yes\",'Source Data'!$I$2:$I$59)"],
  ["=STDEV.S(FILTER('Source Data'!$I$2:$I$59,'Source Data'!$N$2:$N$59=\"Yes\"))"],
  ["=MINIFS('Source Data'!$I$2:$I$59,'Source Data'!$N$2:$N$59,\"Yes\")"],
  ["=MAXIFS('Source Data'!$I$2:$I$59,'Source Data'!$N$2:$N$59,\"Yes\")"],
  ["=AVERAGEIFS('Source Data'!$I$2:$I$59,'Source Data'!$A$2:$A$59,\"618\",'Source Data'!$B$2:$B$59,12)"],
];
summarySheet.getRange("B4:B12").format.numberFormat = "0";
summarySheet.getRange("B13:B14").format.numberFormat = "0";
summarySheet.getRange("B15:B17").format.numberFormat = "0";
summarySheet.getRange("A8").format.wrapText = true;
summarySheet.getRange("A8:B8").format.rowHeight = 30;
summarySheet.getRange("A3:B17").format.borders = {
  insideHorizontal: { style: "thin", color: border },
  bottom: { style: "thin", color: border },
};
summarySheet.getRange("A3:B3").format = {
  fill: headerFill,
  font: { bold: true, color: ink, size: 10 },
};

summarySheet.getRange("A20:G20").merge();
summarySheet.getRange("A20").values = [["Completed manuscript sentence"]];
summarySheet.getRange("A20:G20").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF", size: 10 },
};
summarySheet.getRange("A21:G23").merge();
summarySheet.getRange("A21").values = [[manuscriptSentence]];
summarySheet.getRange("A21:G23").format = {
  fill: paleFill,
  font: { color: ink, size: 10 },
  wrapText: true,
  verticalAlignment: "center",
  borders: { preset: "outside", style: "thin", color: border },
};

summarySheet.getRange("A26:G26").merge();
summarySheet.getRange("A26").values = [["Explicit exception and exclusion"]];
summarySheet.getRange("A26:G26").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF", size: 10 },
};
summarySheet.getRange("A27:G29").values = [
  ["618 exception", excluded180Pair, "180 d", "Shown under 12-month return; excluded from 90 +/- 4 d summary", null, null, null],
  ["1616 short interval", excludedPair, "14 d", "Shown under Other intervals; excluded from 90 +/- 4 d summary", null, null, null],
  ["Retained 1616 interval", retained1616Pair, "110 d", "Shown under 6-month return; included in 90 +/- 4 d summary", null, null, null],
];
summarySheet.getRange("A27:A29").format.font = { bold: true, color: ink, size: 9 };
summarySheet.getRange("A27:G29").format.fill = warningFill;
summarySheet.getRange("D28:G28").merge();
summarySheet.getRange("D28:G28").format.wrapText = true;
summarySheet.getRange("A28:G28").format.rowHeight = 36;

summarySheet.getRange("A32:G32").merge();
summarySheet.getRange("A32").values = [["Provenance"]];
summarySheet.getRange("A32:G32").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF", size: 10 },
};
const provenanceRows = [
  ["Comprehensive ledger", path.basename(sourcePath), sha256(sourceText)],
  ["Ledger manifest", path.basename(sourceManifestPath), sha256(sourceManifestText)],
  ...sourceManifest.source_files.map((source) => ["Upstream source", source.path, source.sha256]),
];
summarySheet.getRange(`A33:C${32 + provenanceRows.length}`).values = provenanceRows;
summarySheet.getRange(`A33:C${32 + provenanceRows.length}`).format = {
  font: { color: muted, size: 9 },
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: border } },
};
summarySheet.getRange(`A33:A${32 + provenanceRows.length}`).format.font = {
  bold: true,
  color: ink,
  size: 9,
};
summarySheet.getRange("A1:A40").format.columnWidth = 34;
summarySheet.getRange("B1:B40").format.columnWidth = 31;
summarySheet.getRange("C1:C40").format.columnWidth = 48;
summarySheet.getRange("D1:G40").format.columnWidth = 18;
summarySheet.getRange("A21:G23").format.rowHeight = 24;
summarySheet.getRange(`A33:C${32 + provenanceRows.length}`).format.rowHeight = 34;
summarySheet.freezePanes.freezeRows(1);

workbook.recalculate();
const publicationInspection = await workbook.inspect({
  kind: "table",
  range: "'Supplementary Table 2'!A1:G27",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 8,
  maxChars: 12000,
});
const summaryInspection = await workbook.inspect({
  kind: "table",
  range: "'Summary and provenance'!A1:G39",
  include: "values,formulas",
  tableMaxRows: 40,
  tableMaxCols: 8,
  maxChars: 12000,
});
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
  maxChars: 6000,
});
console.log(publicationInspection.ndjson);
console.log(summaryInspection.ndjson);
console.log(formulaErrors.ndjson);

const previews = [
  { sheetName: "Supplementary Table 2", range: "A1:G27", fileName: "preview_table.png", scale: 1.25 },
  { sheetName: "Source Data", range: "A1:N59", fileName: "preview_source.png", scale: 0.65 },
  { sheetName: "Summary and provenance", range: "A1:G39", fileName: "preview_summary.png", scale: 1.0 },
];
for (const preview of previews) {
  const rendered = await workbook.render({
    sheetName: preview.sheetName,
    range: preview.range,
    scale: preview.scale,
    format: "png",
  });
  await fs.writeFile(path.join(outputRoot, preview.fileName), new Uint8Array(await rendered.arrayBuffer()));
}

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(workbookPath);
await fs.writeFile(companionCsvPath, sourceText, "utf8");
await fs.writeFile(sentencePath, `${manuscriptSentence}\n`, "utf8");
await fs.writeFile(
  inspectionPath,
  [publicationInspection.ndjson, summaryInspection.ndjson, formulaErrors.ndjson].join("\n"),
  "utf8",
);

const finalManifest = {
  artifact: "Excel companion to final Supplementary Table 2 PDF",
  workbook: path.basename(workbookPath),
  workbook_sheets: ["Supplementary Table 2", "Source Data", "Summary and provenance"],
  source_csv: path.basename(sourcePath),
  source_csv_sha256: sha256(sourceText),
  displayed_intervals: sourceRows.length,
  displayed_mice: animals.length,
  behavior_intervals: behaviorRows.length,
  behavior_mice: new Set(behaviorRows.map((row) => row["Animal ID"])).size,
  maintained_calcium_intervals: sourceRows.filter((row) =>
    parseBoolean(row["Maintained calcium imaging"]),
  ).length,
  behavior_and_calcium_intervals: overlapRows.length,
  behavior_only_intervals: behaviorRows.length - overlapRows.length,
  approximately_3_month_summary: {
    n: approximatelyThreeMonthValues.length,
    mean_days: mean(approximatelyThreeMonthValues),
    sample_sd_days: sampleSd(approximatelyThreeMonthValues),
    minimum_days: Math.min(...approximatelyThreeMonthValues),
    maximum_days: Math.max(...approximatelyThreeMonthValues),
  },
  explicit_exception_days: 180,
  short_pair_excluded_from_fixed_month_and_long_gap_summaries: excludedPair,
  approximately_3_month_excluded_pairs: [excludedPair, excluded180Pair],
  display_month_assignments: {
    [retained1616Pair]: 6,
    [excluded180Pair]: 12,
  },
  manuscript_sentence: manuscriptSentence,
};
await fs.writeFile(manifestPath, `${JSON.stringify(finalManifest, null, 2)}\n`, "utf8");
await fs.writeFile(
  validationPath,
  `# Supplementary Table 2 workbook validation\n\n- First sheet matches the final one-row-per-mouse PDF table and contains no session identifiers.\n- Source Data contains all 58 displayed intervals from 18 mice.\n- Behavior intervals: 56 from 17 mice; 35 overlap maintained calcium imaging and 21 are behavior only.\n- Calcium-imaged intervals: 37 (36 long-gap and one short), including mouse 1616's 14- and 110-d intervals.\n- The reader-facing month columns contain 57 intervals: mouse 1616's retained 110-d interval is shown under 6 months, and mouse 618's 180-d interval remains under 12 months.\n- Approximately 3-month summary: 56 intervals, ${mean(approximatelyThreeMonthValues).toFixed(4)} +/- ${sampleSd(approximatelyThreeMonthValues).toFixed(4)} d, range ${Math.min(...approximatelyThreeMonthValues).toFixed(4)}-${Math.max(...approximatelyThreeMonthValues).toFixed(4)} d.\n- The calculation excludes ${excludedPair} (14 d) and ${excluded180Pair} (180 d).\n- Formula-error scan and rendered visual previews cover all three worksheets.\n`,
  "utf8",
);

console.log(
  JSON.stringify(
    {
      workbookPath,
      sheets: finalManifest.workbook_sheets,
      publicationRows: matrixRows.length,
      sourceRows: sourceRows.length,
      approximatelyThreeMonthSummary: finalManifest.approximately_3_month_summary,
    },
    null,
    2,
  ),
);

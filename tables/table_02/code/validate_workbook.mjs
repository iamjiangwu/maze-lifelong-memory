import fs from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dependencyRoot = process.env.ARTIFACT_TOOL_NODE_MODULES;
if (!dependencyRoot) {
  throw new Error("Set ARTIFACT_TOOL_NODE_MODULES to the dependency runtime's node_modules directory.");
}
const require = createRequire(path.join(dependencyRoot, "__artifact_tool_loader__.cjs"));
const { FileBlob, SpreadsheetFile } = require("@oai/artifact-tool");

const here = path.dirname(fileURLToPath(import.meta.url));
const workbookPath = path.resolve(
  process.argv[2] ?? path.join(here, "..", "build", "Supplementary_Table_2_Long_Gap_Intervals.xlsx"),
);
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));

const publication = await workbook.inspect({
  kind: "table",
  range: "'Supplementary Table 2'!A1:G27",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 8,
  maxChars: 12000,
});
const summary = await workbook.inspect({
  kind: "table",
  range: "'Summary and provenance'!A3:F18",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 8,
  maxChars: 12000,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "post-export formula error scan",
  maxChars: 5000,
});
const workbookInfo = await fs.stat(workbookPath);
const publicationValues = workbook.worksheets
  .getItem("Supplementary Table 2")
  .getRange("A1:G27").values;
const summaryValues = workbook.worksheets
  .getItem("Summary and provenance")
  .getRange("A3:B17").values;
const sourceValues = workbook.worksheets.getItem("Source Data").getRange("A1:N59").values;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

assert(publicationValues[3][0] === "418", "Publication row for mouse 418 is missing.");
assert(Number(publicationValues[3][4]).toFixed(1) === "90.0", "Mouse 418 12-month value mismatch.");
const mouse618 = publicationValues.find((row) => row[0] === "618");
const mouse1616 = publicationValues.find((row) => row[0] === "1616");
assert(mouse618, "Publication row for mouse 618 is missing.");
assert(Number(mouse618[4]).toFixed(1) === "180.0", "Mouse 618 must retain 180 d under 12 months.");
assert(mouse1616, "Publication row for mouse 1616 is missing.");
assert(Number(mouse1616[2]).toFixed(1) === "110.0", "Mouse 1616 must show 110 d under 6 months.");
assert(Number(mouse1616[5]).toFixed(1) === "14.0", "Mouse 1616 must show 14 d under Other intervals.");
assert(mouse1616[6] === "6; other interval", "Mouse 1616 calcium-return label mismatch.");
assert(summaryValues[1][1] === 58, "Displayed interval total mismatch.");
assert(summaryValues[3][1] === 56, "Behavior interval total mismatch.");
assert(summaryValues[5][1] === 37, "Maintained calcium total mismatch.");
assert(summaryValues[9][1] === 56, "Approximately 3-month interval total mismatch.");
assert(Number(summaryValues[10][1]).toFixed(1) === "90.1", "Mean interval mismatch.");
assert(Number(summaryValues[11][1]).toFixed(1) === "3.6", "Sample s.d. mismatch.");
assert(sourceValues.length === 59, "Source Data row count mismatch.");
assert(sourceValues.filter(row => row[11] === "1616day1->1616bday1").length === 1, "Missing/duplicate short pair.");
const short1616 = sourceValues.find(row => row[11] === "1616day1->1616bday1");
const retained1616 = sourceValues.find(row => row[11] === "1616bday1->1616b-3day1");
const exception618 = sourceValues.find(row => row[11] === "618-6day1->618-12day1");
assert(short1616[8] === 14 && short1616[1] === null && short1616[13] === "No", "Short interval summary exclusion mismatch.");
assert(retained1616[8] === 110 && retained1616[1] === 6 && retained1616[13] === "Yes", "Retained 1616 assignment mismatch.");
assert(exception618[8] === 180 && exception618[1] === 12 && exception618[13] === "No", "Mouse 618 assignment/exclusion mismatch.");
assert(publicationValues[24][0].includes("excluded mouse 1616's 14-d interval"), "Footnote omits the 1616 exclusion.");
assert(publicationValues[24][0].includes("mouse 618's 180-d interval"), "Footnote omits the 618 exclusion.");

for (let index = 1; index < sourceValues.length; index += 1) {
  const row = sourceValues[index];
  assert(Number.isInteger(row[8]), `Noninteger interval at source row ${index + 1}`);
  assert(row[8] === Number(row[7]) - Number(row[5]), `Date difference mismatch at source row ${index + 1}`);
}
const pubRows = publicationValues.slice(3, 21);
for (const row of sourceValues.slice(1)) {
  const pub = pubRows.find((candidate) => candidate[0] === row[0]);
  const monthColumn = [3, 6, 9, 12].indexOf(row[1]) + 1;
  const column = monthColumn > 0 ? monthColumn : 5;
  assert(pub[column] === row[8], `Matrix mismatch for ${row[11]}`);
}
assert(!/"kind":"match"/.test(errors.ndjson) || !errors.ndjson.includes('"value":"#'), "Formula errors found");
console.log(publication.ndjson);
console.log(summary.ndjson);
console.log(errors.ndjson);
console.log(
  JSON.stringify(
    {
      workbook_bytes: workbookInfo.size,
      publication_rows: 18,
      source_rows: 58,
      post_export_formula_errors: 0,
    },
    null,
    2,
  ),
);

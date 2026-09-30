import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(path.join(process.env.ARTIFACT_TOOL_NODE_MODULES, '__artifact_loader__.cjs'));
const { FileBlob, SpreadsheetFile } = require('@oai/artifact-tool');
const [workbookPath, outputDir, prefix] = process.argv.slice(2);
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
await fs.mkdir(outputDir, {recursive:true});
for (const [name, sheetName, range] of [
  ['publication', 'Supplementary Table 2', 'A1:G27'],
  ['short_note', 'Source Data', 'L56:N59'],
]) {
  const preview = await workbook.render({sheetName, range, scale:1.5, format:'png'});
  await fs.writeFile(path.join(outputDir, `${prefix}_${name}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const result = await workbook.inspect({kind:'table',range:"'Supplementary Table 2'!A23:G26",include:'values,formulas',tableMaxRows:4,tableMaxCols:7,maxChars:6000});
await fs.writeFile(path.join(outputDir, `${prefix}_reference_cells.ndjson`),result.ndjson);
console.log(JSON.stringify({status:'PASS',workbook:workbookPath,outputDir,prefix}));

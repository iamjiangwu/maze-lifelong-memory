import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const require = createRequire(import.meta.url);
const {FileBlob, SpreadsheetFile, Workbook} = require('@oai/artifact-tool');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.resolve(process.argv[2] ?? path.join(root, 'build'));
const source = JSON.parse(await fs.readFile(path.join(root, 'source_data', 'current_rows.json'), 'utf8'));
// Recreate the existing table style without the inherited template's stale merge-render state.
const workbook = Workbook.create();
const sheet = workbook.worksheets.add('Table S1');
for (const [column,width] of [['A',42.5],['B',29.375],['C',45],['D',62.5]]) sheet.getRange(`${column}:${column}`).format.columnWidth=width;
sheet.getRange('A1:D1').merge();
sheet.getRange('A1:D1').format.font={name:'Arial',size:14,bold:true,color:'#111111'};
sheet.getRange('A1:D1').format.rowHeight=25.5;
sheet.getRange('A2:D2').format.rowHeight=7.5;
sheet.getRange('A3:D3').format.font={name:'Arial',size:8,bold:true,color:'#222222'};
sheet.getRange('A3:D3').format.fill='#E4E4E4';
sheet.getRange('A3:D3').format.rowHeight=21;
sheet.getRange('A3:D3').format.borders={preset:'all',style:'thin',color:'#C8C8C8'};
sheet.getRange('A1').values = [[source.title]];
sheet.getRange('A3:D3').values = [source.headers];
const end = source.rows.length + 3;
const body = sheet.getRange(`A4:D${end}`);
body.values = source.rows.map(r => [r.analysis,r.panels,`n = ${r.mice.length}\n${r.mice.join(', ')}`,r.note]);
body.format.font = {name:'Arial',size:8,color:'#222222',bold:false,italic:false};
body.format.wrapText = true;
body.format.verticalAlignment = 'top';
body.format.borders = {preset:'all',style:'thin',color:'#C8C8C8'};
sheet.showGridLines = false;
const merges=[];
for (const [column,key] of [['C','mice']]) {
  let first=0;
  while(first<source.rows.length) {
    let last=first;
    const val=JSON.stringify(source.rows[first][key]);
    while(last+1<source.rows.length && JSON.stringify(source.rows[last+1][key])===val) last++;
    if(last>first) {
      sheet.getRange(`${column}${first+5}:${column}${last+4}`).clear({applyTo:'contents'});
      const address=`${column}${first+4}:${column}${last+4}`;
      sheet.getRange(address).merge();
      sheet.getRange(address).format.verticalAlignment='center';
      merges.push(address);
    }
    first=last+1;
  }
}
for(let index=0;index<source.rows.length;index++) {
  const r=source.rows[index];
  const wrapCount=(s,n)=>String(s).split('\n').reduce((total,line)=>total+Math.ceil(line.length/n),0);
  const sharedMice = merges.some(address => {const [first,last]=address.match(/\d+/g).map(Number);return index+4>=first&&index+4<=last;});
  const nLines=Math.max(wrapCount(r.analysis,38),wrapCount(r.panels,25),wrapCount(r.note,59),sharedMice?0:wrapCount(r.mice.join(', '),40)+1);
  sheet.getRange(`A${index+4}:D${index+4}`).format.rowHeight=Math.max(31,nLines*10.5+8);
}
workbook.recalculate();
const errorScan=await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A',options:{useRegex:true,maxResults:100},maxChars:5000});
if(errorScan.ndjson.includes('"kind":"match"')) throw new Error(errorScan.ndjson);
await fs.mkdir(output,{recursive:true});
const workbookPath=path.join(output,'Supplementary_Table_S1_Mouse_Cohorts_by_Analysis.xlsx');
const result=await SpreadsheetFile.exportXlsx(workbook);
await result.save(workbookPath);
const renderedWorkbook=await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
for(const [start,stop,label] of [[1,end,'full']]) {
  const preview=await renderedWorkbook.render({sheetName:'Table S1',range:`A${start}:D${stop}`,scale:1,format:'png'});
  await fs.writeFile(path.join(output,`Supplementary_Table_S1_workbook_${label}.png`),new Uint8Array(await preview.arrayBuffer()));
}
await fs.writeFile(path.join(output,'WORKBOOK_BUILD.json'),JSON.stringify({status:'PASS',rows:source.rows.length,merges,formula_errors:0},null,2)+'\n');
console.log(JSON.stringify({status:'PASS',rows:source.rows.length,output}));

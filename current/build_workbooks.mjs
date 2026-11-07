import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const root=path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/,'$1');
const books=JSON.parse(await fs.readFile(path.join(root,'WORKBOOK_TABLES.json'),'utf8'));
const out=path.join(root,'deliverables','Supplementary_Data');
const previews=path.join(root,'workbook_previews');
await fs.mkdir(out,{recursive:true});await fs.mkdir(previews,{recursive:true});
const ledger=[];
const selectedBooks=new Set(process.argv.slice(2));
for(const [number,sheets] of Object.entries(books)){
  if(selectedBooks.size && !selectedBooks.has(number))continue;
  const wb=Workbook.create();const names=['README','Definitions',...Object.keys(sheets).filter(n=>!['README','Definitions'].includes(n))];
  for(const name of names){
    const records=sheets[name];if(!records.length)throw new Error(`Empty ${number}/${name}`);
    const fields=[...new Set(records.flatMap(r=>Object.keys(r)))];
    const rows=[fields,...records.map(r=>fields.map(k=>r[k]??'NA'))];
    const sh=wb.worksheets.add(name);sh.showGridLines=false;
    sh.getRangeByIndexes(0,0,rows.length,fields.length).values=rows;
    const range=sh.getRangeByIndexes(0,0,rows.length,fields.length);
    range.format.font={name:'Arial',size:10};range.format.rowHeight=17;
    range.format.columnWidth=17;
    sh.getRangeByIndexes(0,0,1,fields.length).format={fill:'#16324F',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:44};
    sh.freezePanes.freezeRows(1);sh.freezePanes.freezeColumns(Math.min(2,fields.length));
    if(['README','Definitions'].includes(name)){
      sh.getRangeByIndexes(0,0,rows.length,1).format.columnWidth=24;
      sh.getRangeByIndexes(0,1,rows.length,1).format.columnWidth=100;
      range.format.wrapText=true;range.format.rowHeight=48;
    }else{
      let lineCount=1;
      for(let j=0;j<fields.length;j++){
        const strings=records.map(r=>r[fields[j]]).filter(v=>typeof v==='string');
        const longest=Math.max(fields[j].length,...strings.map(v=>v.length));
        const width=Math.min(fields[j]==='configuration'?80:60,Math.max(14,longest+2));
        sh.getRangeByIndexes(0,j,rows.length,1).format.columnWidth=width;
        if(strings.length){
          sh.getRangeByIndexes(1,j,records.length,1).format.wrapText=true;
          lineCount=Math.max(lineCount,Math.ceil(longest/Math.max(10,width-4)));
        }
        const values=records.map(r=>r[fields[j]]).filter(v=>typeof v==='number');
        if(values.length){
          const integer=values.every(Number.isInteger);sh.getRangeByIndexes(1,j,records.length,1).setNumberFormat(integer?'0':'0.0000');
        }
      }
      if(lineCount>1)sh.getRangeByIndexes(1,0,records.length,fields.length).format.rowHeight=Math.min(90,17*lineCount);
    }
    const lastCol=Math.min(fields.length,8);let col='';let v=lastCol;while(v){v--;col=String.fromCharCode(65+v%26)+col;v=Math.floor(v/26);}
    const img=await wb.render({sheetName:name,range:`A1:${col}${Math.min(rows.length,9)}`,scale:1.1,format:'png'});
    const safe=name.replace(/[^a-zA-Z0-9_-]/g,'_');await fs.writeFile(path.join(previews,`${number}_${safe}.png`),new Uint8Array(await img.arrayBuffer()));
    ledger.push({workbook:Number(number),sheet:name,rows:records.length,columns:fields.length,preview:`${number}_${safe}.png`});
  }
  wb.recalculate();
  const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(path.join(out,`Supplementary_Data_${number}.xlsx`));
  console.log(`WORKBOOK_COMPLETE ${number} sheets=${names.length}`);
}
if(selectedBooks.size){
  const previous=JSON.parse(await fs.readFile(path.join(root,'WORKBOOK_BUILD.json'),'utf8'));
  ledger.push(...previous.tables.filter(t=>!selectedBooks.has(String(t.workbook))));
  ledger.sort((a,b)=>a.workbook-b.workbook);
}
await fs.writeFile(path.join(root,'WORKBOOK_BUILD.json'),JSON.stringify({engine:'@oai/artifact-tool',tables:ledger},null,2));

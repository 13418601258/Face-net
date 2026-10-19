"""Verify final HTML encoding, scope, links, and evidence freshness."""
from pathlib import Path
from html.parser import HTMLParser
import json,hashlib

ROOT=Path(__file__).resolve().parent.parent
HTML=ROOT/'log/方向/实验计划-阶段0至3.html'
OUT=ROOT/'code/reports/stage3_s36_v5_final'

class Check(HTMLParser):
    tags={'div','details','section','table','thead','tbody','tr','td','th','ol','ul','li','p'}
    def __init__(self):super().__init__();self.stack=[];self.errors=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag in self.tags:self.stack.append(tag)
        if tag=='a':self.links.extend(v for k,v in attrs if k=='href')
    def handle_endtag(self,tag):
        if tag in self.tags:
            if not self.stack or self.stack[-1]!=tag:self.errors.append([tag,self.stack[-3:]])
            else:self.stack.pop()

def main():
    raw=HTML.read_bytes();text=raw.decode('utf-8');before=(OUT/'s36_before_final.html').read_bytes().decode('utf-8-sig')
    anchor='<div class="step" data-id="s36">';a=text.index(anchor);b=text.index('        </section>',a)
    a0=before.index(anchor);b0=before.index('        </section>',a0)
    parser=Check();parser.feed(text[a:b]);missing=[link for link in parser.links if not (HTML.parent/link).resolve().exists()]
    files=[HTML,ROOT/'code/代码文件说明.txt',OUT/'final_report.md',OUT/'summary.json']
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    checks=dict(utf8_no_bom=not raw.startswith(bytes([239,187,191])),
                no_replacement_characters=all(chr(65533) not in p.read_text(encoding='utf-8') for p in files),
                outside_s36_unchanged=text[:a]==before[:a0] and text[b:]==before[b0:],
                s36_balanced=not parser.errors and not parser.stack,all_active_links_exist=not missing,
                final_stop_written='不进入第二关' in text[a:b],
                scientific_outputs_unchanged=all(sha(OUT/p)==digest for p,digest in manifest['files'].items()))
    result=dict(passed=all(checks.values()),checks=checks,missing_links=missing,parse_errors=parser.errors,
                files={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    (OUT/'delivery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()

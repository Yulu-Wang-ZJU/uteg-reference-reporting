"""Compact atomic JSON serialization; retains every field and numerical value."""
import json
def install(module):
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.name+'.tmp')
        temp.write_text(json.dumps(module.safe(value),ensure_ascii=False,separators=(',',':')),encoding='utf8')
        temp.replace(path)
    module.write=write

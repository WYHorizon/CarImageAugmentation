"""Reuse existing v5 packed geometry with two or three switchable textures."""
import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import re

from PIL import Image


def sha_bytes(value):return hashlib.sha256(value).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-preview',type=Path,required=True)
    p.add_argument('--original-texture',type=Path,required=True)
    p.add_argument('--clean-texture',type=Path,required=True)
    p.add_argument('--strong-texture',type=Path,
                   help='Optional third texture; omit to retain the original two-choice preview')
    p.add_argument('--default-mode',choices=['original','clean','strong'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--record',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.record.exists():raise ValueError('Refusing to overwrite an existing preview or record')
    texture_paths=[('original',a.original_texture),('clean',a.clean_texture)]
    if a.strong_texture:texture_paths.append(('strong',a.strong_texture))
    mode=a.default_mode or ('strong' if a.strong_texture else 'clean')
    if mode not in dict(texture_paths):raise ValueError('Requested default texture was not supplied')
    labels={'original':'原 v5 贴图','clean':'轻度去反光','strong':'重度去反光'}
    root_id='bmw5-strong-texture' if a.strong_texture else 'bmw5-clean-texture'
    viewer=dict(defaultMode=mode,labels={k:labels[k] for k,_ in texture_paths},
                stateKey='bmw5_strong_texture' if a.strong_texture else 'bmw5_clean_texture',
                previewId='bmw5_v5_qwen_strength_textures' if a.strong_texture else 'bmw5_v5_qwen_mild_texture')
    old=a.original_preview.read_text()
    scripts=list(re.finditer(r'<script([^>]*)>([\s\S]*?)</script>',old))
    data=json.loads(scripts[0].group(2));bundle=scripts[1].group(2)
    original_geometry={k:v for k,v in data.items() if k!='texture'}
    template=Path(__file__).with_name('texture_preview_template.html').read_text()
    options=''.join('<option value="'+k+'"'+(' selected' if k==mode else '')+'>'+html.escape(labels[k])+'</option>'
                    for k,_ in ([('strong',a.strong_texture),('clean',a.clean_texture),('original',a.original_texture)] if a.strong_texture else [('clean',a.clean_texture),('original',a.original_texture)]))
    template=template.replace('__ROOT_ID__',root_id).replace('__MODE_OPTIONS__',options)
    template=template.replace('__PREVIEW_TITLE__','宝马 5 系 · 去反光强度对照' if a.strong_texture else '宝马 5 系 · 轻度去反光纹理')
    template=template.replace('__STAGE_LABEL__','同一宝马5系模型，可拖拽旋转并切换原 v5、轻度与重度去反光纹理' if a.strong_texture else '同一宝马5系模型，可拖拽旋转并切换原始与轻度去反光纹理')
    for size,quality in [(1024,80),(1024,70),(768,80),(768,65),(640,80),(640,65),(512,80),(512,65)]:
        textures={}
        for key,path in texture_paths:
            image=Image.open(path).convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
            buffer=io.BytesIO();image.save(buffer,format='WEBP',quality=quality,method=6)
            textures[key]='data:image/webp;base64,'+base64.b64encode(buffer.getvalue()).decode()
        packed=dict(original_geometry,textures=textures,viewer=viewer)
        fragment=template.replace('__PACKED_DATA__',json.dumps(packed,separators=(',',':'))).replace('__THREE_BUNDLE__',bundle)
        if len(fragment.encode())<990000:break
    else:raise ValueError('Preview exceeds the inline size budget')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(fragment)
    reread=a.output.read_text()
    payload=json.loads(re.search(r'<script type="application/json" id="clean-mesh-data">([\s\S]*?)</script>',reread).group(1))
    assert {k:v for k,v in payload.items() if k not in {'textures','viewer'}}==original_geometry
    assert not re.search(r'__(?:ROOT_ID|MODE_OPTIONS|PREVIEW_TITLE|STAGE_LABEL|PACKED_DATA|THREE_BUNDLE)__',reread)
    record=dict(method='reuse_exact_v5_packed_preview_geometry_switch_textures',
        source_preview=str(a.original_preview.resolve()),source_preview_sha256=sha_bytes(old.encode()),
        output=str(a.output.resolve()),output_sha256=sha_bytes(fragment.encode()),bytes=len(fragment.encode()),
        geometry_fields_identical=True,packed_geometry_sha256=sha_bytes(base64.b64decode(data['geometry'])),
        vertices=data['vertexCount'],faces=data['faceCount'],
        preview_texture_format='webp',preview_texture_size=size,preview_texture_quality=quality,
        full_texture_sha256={k:sha_bytes(path.read_bytes()) for k,path in texture_paths},
        mode_labels=viewer['labels'],default_mode=mode,root_id=root_id,
        same_compression_settings_for_all_modes=True,
        full_model_modified=False,material='MeshBasicMaterial, no lighting, no tonemapping',
        limitation='Same previously simplified and quantized preview as v5; downloadable GLB retains full geometry and 2048 PNG texture.')
    a.record.parent.mkdir(parents=True,exist_ok=True);a.record.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['bytes','geometry_fields_identical','vertices','faces','preview_texture_size','preview_texture_quality']},indent=2))


if __name__=='__main__':main()

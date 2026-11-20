"""Prepare 2D paint edit masks for all eight BMW5 reference photographs.

The reviewed side_004 v2 masks are copied exactly. Other views retain existing
glass/wheel/lamp/plate annotations and use small internal-detail margins. No
source photo, previous mask, raw model output, UV or 3D file is changed.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from prepare_mask import draw_regions, sha256, smoothstep_distance


HANDLE_POINTS = {
    "side_008": {
        "front_handle": [(755,468),(761,461),(812,462),(820,466),(822,472),(815,479),(768,480),(758,475)],
        "rear_handle": [(456,457),(462,451),(508,450),(518,454),(522,459),(516,466),(507,470),(469,469),(459,465)],
    },
    "front_001": {
        "front_handle": [(1077,442),(1081,436),(1102,434),(1108,438),(1110,452),(1105,458),(1090,460),(1080,455)],
        "rear_handle": [(1219,438),(1223,430),(1237,429),(1243,434),(1244,446),(1235,451),(1224,449)],
    },
    "front_003": {
        "front_handle": [(425,436),(430,431),(449,432),(455,438),(454,449),(443,455),(429,452),(424,446)],
        "rear_handle": [(291,429),(299,423),(315,423),(320,429),(315,439),(306,445),(296,443),(291,438)],
    },
    "rear_005": {
        "front_handle": [(425,451),(431,445),(452,443),(459,447),(457,461),(450,468),(433,470),(426,465)],
        "rear_handle": [(587,438),(594,432),(618,430),(625,432),(627,441),(622,449),(616,455),(596,457),(586,452)],
    },
    "rear_007": {
        "front_handle": [(1079,460),(1087,454),(1102,456),(1110,462),(1111,473),(1105,479),(1094,480),(1083,473)],
        "rear_handle": [(911,441),(917,436),(940,437),(947,443),(953,449),(951,458),(941,463),(925,461),(915,455)],
    },
}


def polyline(name, points, width=2):
    return dict(name=name, kind="line", label=5, points=points, width=width)


def extra_outer_regions(view_id):
    # Source annotation review revealed small tire sidewalls/far mirrors outside
    # the old approximate ellipses. Lock them here without modifying those inputs.
    points = {
        "front_001": [
            ("far_front_tire_complete", [(218,675),(436,677),(426,703),(402,731),(358,746),(278,742),(235,716)]),
            ("near_rear_tire_sidewall", [(1211,654),(1250,654),(1270,714),(1255,724),(1221,713),(1192,672)]),
            ("far_mirror", [(450,375),(461,363),(481,358),(500,364),(505,373),(488,389),(453,389)]),
        ],
        "front_003": [
            ("far_front_tire_complete", [(1049,685),(1315,684),(1304,714),(1265,750),(1186,759),(1119,746),(1073,709)]),
            ("near_rear_tire_sidewall", [(278,534),(301,572),(316,610),(336,653),(336,681),(316,709),(284,717),(285,661),(285,596)]),
            ("far_mirror", [(1032,367),(1059,367),(1080,377),(1087,391),(1079,405),(1043,396)]),
        ],
    }
    return [dict(name=name,kind="polygon",label=5,points=p) for name,p in points.get(view_id,[])]


def is_internal(annotation):
    name = annotation["name"]
    return any(key in name for key in ("handle", "seam", "rocker", "fuel_cap", "model_text")) or name in {"hood_badge", "rear_badge"}


def refined_features(view_id, original, side_manifest):
    entries = []
    for region in original:
        if not is_internal(region):
            continue
        entry = copy.deepcopy(region)
        if entry["name"] in HANDLE_POINTS.get(view_id, {}):
            entry["kind"] = "polygon"
            entry["points"] = HANDLE_POINTS[view_id][entry["name"]]
        elif view_id == "front_003" and entry["name"] == "hood_badge":
            entry["box"] = [1097,462,1133,488]
        elif entry["kind"] == "line":
            entry["width"] = 2
        entries.append(entry)
    if view_id == "side_008":
        # Mirror only the reviewed thin hood/bumper/rocker outlines; account for
        # the separately photographed right-side door positions below.
        keep = {"visible_hood_panel_seam", "front_bumper_fender_join", "rear_bumper_quarter_join",
                "rocker_upper_seam_refined", "rocker_lower_edge_refined"}
        entries = [e for e in entries if e["name"] != "rocker_trim"]
        for old in side_manifest["manual_protection"]:
            if old["name"] in keep:
                entry = copy.deepcopy(old)
                entry["points"] = [(1535-x,y) for x,y in old["points"]]
                entries.append(entry)
    replacements = {
        "side_008": {
            "front_door_seam": [(1022,442),(1032,452),(1037,470),(1039,495),(1039,540),(1038,585),(1035,632)],
            "middle_door_seam": [(712,433),(719,453),(725,480),(729,515),(731,553),(730,590),(726,628)],
            "rear_door_seam": [(385,415),(400,443),(421,476),(450,520),(481,568),(514,624)],
            "fuel_cap": [(306,422),(350,421),(363,425),(370,443),(372,458),(367,464),(310,466),(302,461),(296,435),(298,426),(306,422)],
        },
        "front_001": {
            "middle_door_seam": [(1102,406),(1114,442),(1122,480),(1126,529),(1126,577),(1123,629)],
            "rear_door_seam": [(1245,395),(1256,426),(1259,462),(1254,493),(1244,524)],
            "rocker_trim": [(928,652),(1013,644),(1121,635),(1222,621)],
        },
        "front_003": {
            "middle_door_seam": [(419,402),(413,440),(408,478),(407,519),(410,563),(419,626)],
            "rear_door_seam": [(285,389),(275,428),(273,465),(279,500),(288,525)],
            "rocker_trim": [(306,618),(413,631),(503,643),(598,653)],
        },
        "rear_005": {
            "middle_door_seam": [(510,412),(490,445),(477,480),(465,523),(461,565),(465,606),(480,641)],
            "rear_door_seam": [(697,389),(673,418),(647,451),(623,486),(607,520),(594,556),(585,594),(582,640)],
            "rocker_trim": [(335,635),(479,642),(585,645)],
        },
        "rear_007": {
            "middle_door_seam": [(1030,419),(1047,451),(1060,487),(1067,527),(1069,568),(1068,610),(1065,650)],
            "rear_door_seam": [(837,393),(863,427),(889,460),(912,499),(927,538),(934,581),(936,647)],
            "rocker_trim": [(936,649),(1064,651),(1213,646)],
        },
    }
    for entry in entries:
        points = replacements.get(view_id, {}).get(entry["name"])
        if points is not None:
            entry["points"] = points
    additions = {
        "front_001": [polyline("front_door_seam_added", [(949,411),(958,450),(965,493),(969,546),(969,600),(966,647)]),
                      polyline("hood_outer_seam_added", [(567,489),(692,459),(805,435),(895,420),(910,408),(914,391)])],
        "front_003": [polyline("front_door_seam_added", [(584,414),(574,453),(567,499),(564,551),(566,605),(570,649)]),
                      polyline("hood_outer_seam_added", [(619,389),(618,407),(630,420),(758,451),(938,488)])],
    }
    entries.extend(additions.get(view_id, []))
    return entries


def write_view(view_id, rgb_path, labels_path, geometry_path, annotation_path, source_annotations,
               side_manifest, output, parameters):
    output.mkdir(parents=True, exist_ok=True)
    photo = np.asarray(Image.open(rgb_path).convert("RGB"))
    labels = np.asarray(Image.open(labels_path).convert("L"))
    geometry = np.asarray(Image.open(geometry_path).convert("L")) > 127
    if photo.shape != (1024,1536,3) or labels.shape != (1024,1536) or geometry.shape != labels.shape:
        raise ValueError(f"Unexpected size for {view_id}")
    if not set(np.unique(labels)).issubset(set(range(6))):
        raise ValueError(f"Unknown semantic labels in {view_id}")
    coarse = draw_regions([a for a in source_annotations if is_internal(a)])
    retained = draw_regions([a for a in source_annotations if a["label"] == 5 and not is_internal(a)])
    outer = ~geometry | np.isin(labels,[0,2,3,4]) | retained | ((labels == 5) & ~coarse)
    added_outer = extra_outer_regions(view_id)
    outer |= draw_regions(added_outer)
    features = refined_features(view_id, source_annotations, side_manifest)
    feature = draw_regions(features) & geometry
    outer_distance = distance_transform_edt(~outer)
    feature_distance = distance_transform_edt(~feature)
    a_outer = smoothstep_distance(outer_distance, parameters["outer_inset_px"], parameters["outer_feather_px"])
    a_feature = smoothstep_distance(feature_distance, parameters["feature_inset_px"], parameters["feature_feather_px"])
    alpha = np.rint(np.minimum(a_outer,a_feature)*255).astype(np.uint8)
    support = alpha > 0
    assert support.any() and not np.any(support & (outer | feature))
    assert not np.any(support & np.isin(labels,[0,2,3,4]))
    assert not np.any(support & (labels == 5) & ~coarse)
    arrays = {"soft_alpha.png":alpha,"hard_support.png":support.astype(np.uint8)*255,
              "protected_mask.png":(~support).astype(np.uint8)*255,
              "outer_protection_mask.png":outer.astype(np.uint8)*255,
              "feature_protection_mask.png":feature.astype(np.uint8)*255,
              "refined_paint_domain.png":(~outer & ~feature).astype(np.uint8)*255}
    for name,array in arrays.items(): Image.fromarray(array).save(output/name)
    overlay = photo.astype(np.float32)
    blend = alpha[...,None].astype(np.float32)/255*.44
    overlay = overlay*(1-blend)+np.array([30,220,145])*blend
    overlay[feature] = .3*overlay[feature]+.7*np.array([235,80,175])
    Image.fromarray(np.rint(overlay).astype(np.uint8)).save(output/'overlay.png')
    for name,mask,color in [('outer_protection_overlay.png',outer&geometry,[240,155,50]),
                            ('feature_protection_overlay.png',feature,[235,80,175])]:
        image=photo.astype(np.float32).copy(); image[mask]=image[mask]*.25+np.array(color)*.75
        Image.fromarray(np.rint(image).astype(np.uint8)).save(output/name)
    inputs={"source_rgb":rgb_path,"material_ids":labels_path,"geometry_mask":geometry_path,"reference_annotations":annotation_path}
    a=alpha[...,None].astype(np.float32)/255
    simulation=np.rint(photo*(1-a)+(255-photo)*a).astype(np.uint8)
    assert np.array_equal(simulation[~support],photo[~support])
    record={"status":"complete_pending_visual_review","view_id":view_id,
            "method":"reviewed_outer_material_regions_with_refined_thin_internal_features",
            "script_sha256":sha256(__file__),"mask_helper_sha256":sha256(Path(__file__).with_name('prepare_mask.py')),
            "inputs":{k:{"path":str(p.resolve()),"sha256":sha256(p)} for k,p in inputs.items()},
            "size_wh":[1536,1024],"parameters":parameters,"manual_protection":features,
            "additional_outer_protection":added_outer,
            "replaced_coarse_annotations":[a['name'] for a in source_annotations if is_internal(a)],
            "counts":{"geometry_pixels":int(geometry.sum()),"editable_support_pixels":int(support.sum()),
                      "fully_editable_alpha255_pixels":int((alpha==255).sum()),"locked_original_pixels":int((~support).sum())},
            "contracts":{"soft_alpha":"8-bit L /255; composite edited*alpha + original*(1-alpha), explicitly restore original where alpha==0.",
                         "protected_mask":"255 iff alpha==0, over the entire original image.",
                         "outer_protection":"Glass, lamps, plates, wheels, grille, silhouette and exterior trim remain protected with an 8px inset and 8px feather.",
                         "feature_protection":"Actual handles, narrow panel seams, badges/model lettering and fuel-door outlines use 0px inset, 3px feather; no high-frequency reflection classifier.",
                         "source_coordinates":"No crop, warp or rescale; every mask is aligned to the original prepared RGB."},
            "checks":{"support_excludes_unknown_glass_rubber_alloy":True,"support_excludes_outer_and_refined_features":True,
                      "new_class5_support_only_in_replaced_coarse_features":True,"synthetic_edit_preserves_locked_pixels_exact":True},
            "limitations":["Hand annotations remain approximate; light cleanup and visual review are required.",
                           "Masks preserve known features but do not measure physical reflectance or recover unseen paint."]}
    record['outputs']={p.name:{'sha256':sha256(p)} for p in output.glob('*.png')}
    (output/'manifest.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root',type=Path,required=True)
    parser.add_argument('--regions-root',type=Path,required=True)
    parser.add_argument('--side-mask-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    annotations_path=args.regions_root/'annotations.json'
    annotations=json.loads(annotations_path.read_text())['annotations']
    prepared=json.loads((args.prepared_root/'manifest.json').read_text())
    side_manifest=json.loads((args.side_mask_root/'manifest.json').read_text())
    parameters={'outer_inset_px':8.,'outer_feather_px':8.,'feature_inset_px':0.,'feature_feather_px':3.}
    records=[]
    for view in prepared['views']:
        vid=view['id'];out=args.output_dir/vid
        rgb_path=args.prepared_root/view['files']['source_rgb.png']['path']
        labels_path=args.regions_root/vid/'material_ids.png'
        geometry_path=args.prepared_root/view['files']['mask_geometry.png']['path']
        if vid=='side_004':
            out.mkdir(exist_ok=True)
            record=copy.deepcopy(side_manifest)
            for name,info in record['outputs'].items():
                source=args.side_mask_root/name
                if sha256(source)!=info['sha256']: raise ValueError(f'Reviewed side mask hash changed: {name}')
                shutil.copyfile(source,out/name)
                assert sha256(out/name)==info['sha256']
            assert sha256(rgb_path)==record['inputs']['source_rgb']['sha256']
            record['view_id']=vid;record['reused_from']=str(args.side_mask_root.resolve());record['reused_masks_bitwise_exact']=True
            (out/'manifest.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
        else:
            record=write_view(vid,rgb_path,labels_path,geometry_path,annotations_path,annotations[vid],side_manifest,out,parameters)
        records.append({'id':vid,'source_rgb':record['inputs']['source_rgb'],
                        'soft_alpha':str((out/'soft_alpha.png').resolve()),'protected_mask':str((out/'protected_mask.png').resolve()),
                        'manifest':str((out/'manifest.json').resolve()),'counts':record['counts']})
        print(vid,record['counts']['editable_support_pixels'],flush=True)
    sheet=Image.new('RGB',(1536,4*560),(245,245,245));draw=ImageDraw.Draw(sheet)
    for i,record in enumerate(records):
        image=Image.open(args.output_dir/record['id']/'overlay.png').resize((768,512),Image.Resampling.LANCZOS)
        x=(i%2)*768;y=(i//2)*560;draw.text((x+12,y+12),record['id']+' | green editable / pink precise protected',fill=(30,30,30));sheet.paste(image,(x,y+40))
    sheet.save(args.output_dir/'overlay_contact_sheet.jpg',quality=92)
    manifest={'status':'complete_pending_visual_review','script_sha256':sha256(__file__),
              'size_wh':[1536,1024],'parameters':parameters,'view_count':len(records),'views':records,
              'prepared_manifest_sha256':sha256(args.prepared_root/'manifest.json'),
              'annotations_sha256':sha256(annotations_path),'side004_reused_exact':True}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()

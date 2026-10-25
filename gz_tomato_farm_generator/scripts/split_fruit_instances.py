#!/usr/bin/env python3
"""Post-process generated tomato models so every mesh node is rendered and every tomato is
its own labelled visual.

Usage: split_fruit_instances.py <models_dir> [--only tomato_123,tomato_456]

Why (found 2026-10-11 while building a segmentation-labelled dataset):
- tomato_gen.py names each exported mesh node after its material. Library trusses carry
  duplicated materials (AG15_Fruit1.003, AG15_Leaf1.007, Leaf2.001, ...), but the template
  model.sdf only has visuals for the exact submesh names Branch1/Leaf1/Leaf2/Fruit1-4/
  Blossom1-3, and Gazebo matches <submesh> names exactly. Every other node was silently
  not drawn, including the full-size (46-60 mm) trusses.
- All of a plant's tomatoes sit in a few per-material meshes, so a segmentation camera
  can't tell touching tomatoes apart. Each connected component of a fruit mesh is one
  tomato, so splitting by connectivity gives one visual (and one label) per tomato.

Per plant directory (tomato_*):
- reads meshes/tomato.dae and model.sdf.orig (model.sdf is copied there on first run),
  so reruns always start from the generator's output;
- writes meshes/tomato_instances.dae: non-fruit nodes unchanged, each fruit node replaced
  by one node + compact geometry per tomato, named <node>__tNN;
- writes model.sdf with one visual per node of the new mesh. Materials come from the
  original visual of the same base name (AG15_Leaf1.007 -> leaf1). Each tomato visual gets
  a gz-sim Label plugin with a farm-wide unique label (1..254; 0 is background).
Writes <models_dir>/fruit_instances.json: label -> plant, node, centre (model frame, m),
size (m).
"""
import argparse
import copy
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

C = "http://www.collada.org/2005/11/COLLADASchema"
NS = "{%s}" % C
ET.register_namespace("", C)
NODE_RE = re.compile(r"^(?:AG15_)?(Branch|Leaf|Fruit|Blossom)(\d)(?:\.\d+)?$")
MAX_LABEL = 254
LABEL_PLUGIN = """<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">
          <label>{label}</label>
        </plugin>"""


def base_visual_name(node_name):
    """Name of the template visual whose material this node should use."""
    m = NODE_RE.match(node_name)
    if not m:
        raise ValueError(f"unexpected mesh node name {node_name!r}")
    kind, idx = m.groups()
    return "branch" if kind == "Branch" else f"{kind.lower()}{idx}"


def float_rows(source):
    acc = source.find(f"{NS}technique_common/{NS}accessor")
    stride = int(acc.get("stride", "1"))
    return np.array(source.find(NS + "float_array").text.split(), float).reshape(-1, stride)


def split_components(mesh):
    """Return (triangle index array [n_tri, 3, stride], vertex offset, list of triangle-index
    arrays, one per connected component). Connectivity is over position indices."""
    tris = mesh.findall(NS + "triangles")
    if len(tris) != 1 or mesh.findall(NS + "polylist") or mesh.findall(NS + "polygons"):
        raise ValueError("expected exactly one <triangles> block per fruit mesh")
    tri = tris[0]
    inputs = tri.findall(NS + "input")
    stride = max(int(i.get("offset")) for i in inputs) + 1
    p = np.array(tri.find(NS + "p").text.split(), int).reshape(-1, 3, stride)
    voff = int(next(i for i in inputs if i.get("semantic") == "VERTEX").get("offset"))
    v = p[:, :, voff]
    parent = np.arange(v.max() + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, c in v:
        for u in (b, c):
            ra, ru = find(a), find(u)
            if ra != ru:
                parent[ru] = ra
    roots = np.array([find(a) for a in v[:, 0]])
    comps = [np.nonzero(roots == r)[0] for r in np.unique(roots)]
    return p, voff, comps


def compact_geometry(geom, p, comp, new_id):
    """Copy of `geom` holding only triangles `comp`, with every source cut down to the rows
    those triangles use and indices remapped."""
    g = copy.deepcopy(geom)
    g.set("id", new_id)
    g.set("name", new_id)
    mesh = g.find(NS + "mesh")
    tri = mesh.find(NS + "triangles")
    sub = p[comp]  # [n, 3, stride]
    sources = {s.get("id"): s for s in mesh.findall(NS + "source")}
    vertices = mesh.find(NS + "vertices")
    # sources indexed by each offset (VERTEX resolves through <vertices>)
    by_offset = {}
    for inp in tri.findall(NS + "input"):
        off = int(inp.get("offset"))
        if inp.get("semantic") == "VERTEX":
            ids = [i.get("source")[1:] for i in vertices.findall(NS + "input")]
        else:
            ids = [inp.get("source")[1:]]
        by_offset.setdefault(off, set()).update(ids)
    new_p = sub.copy()
    for off, ids in by_offset.items():
        used, remap = np.unique(sub[:, :, off], return_inverse=True)
        new_p[:, :, off] = remap.reshape(sub.shape[:2])
        for sid in ids:
            src = sources[sid]
            rows = float_rows(src)[used]
            fa = src.find(NS + "float_array")
            fa.text = " ".join(f"{x:.6g}" for x in rows.ravel())
            fa.set("count", str(rows.size))
            src.find(f"{NS}technique_common/{NS}accessor").set("count", str(len(rows)))
    tri.set("count", str(len(comp)))
    tri.find(NS + "p").text = " ".join(map(str, new_p.ravel()))
    # keep ids unique across the file
    for el in g.iter():
        for attr in ("id",):
            if el is not g and el.get(attr):
                el.set(attr, f"{new_id}-{el.get(attr)}")
        if el.get("source", "").startswith("#") and el is not g:
            el.set("source", f"#{new_id}-{el.get('source')[1:]}")
    return g, sub


def process_plant(plant_dir, next_label, instances):
    sdf_path = plant_dir / "model.sdf"
    orig_sdf = plant_dir / "model.sdf.orig"
    if not orig_sdf.exists():
        shutil.copy2(sdf_path, orig_sdf)
    dae_in = plant_dir / "meshes" / "tomato.dae"
    dae_out = plant_dir / "meshes" / "tomato_instances.dae"

    tree = ET.parse(dae_in)
    root = tree.getroot()
    lib_geom = root.find(NS + "library_geometries")
    geoms = {g.get("id"): g for g in lib_geom}
    scene = root.find(f"{NS}library_visual_scenes/{NS}visual_scene")

    visuals = []  # (node_name, base_visual, label or None)
    for node in list(scene):
        if node.tag != NS + "node":
            continue
        ig = node.find(NS + "instance_geometry")
        if ig is None:
            continue
        name = node.get("name")
        base = base_visual_name(name)
        if not base.startswith("fruit"):
            visuals.append((name, base, None))
            continue
        geom = geoms[ig.get("url")[1:]]
        p, voff, comps = split_components(geom.find(NS + "mesh"))
        M = np.array(node.findtext(NS + "matrix").split(), float).reshape(4, 4)
        positions = None
        mesh = geom.find(NS + "mesh")
        vsrc = mesh.find(NS + "vertices").find(NS + "input").get("source")[1:]
        positions = float_rows(next(s for s in mesh.findall(NS + "source") if s.get("id") == vsrc))
        # order tomatoes bottom-up so labels are stable and readable
        comps.sort(key=lambda c: positions[np.unique(p[c][:, :, voff])][:, 2].mean())
        for k, comp in enumerate(comps):
            safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
            gid = f"{safe}__t{k:02d}-mesh"
            g, sub = compact_geometry(geom, p, comp, gid)
            lib_geom.append(g)
            n = copy.deepcopy(node)
            n_name = f"{name}__t{k:02d}"
            n.set("id", f"{safe}__t{k:02d}")
            n.set("name", n_name)
            n.find(NS + "instance_geometry").set("url", "#" + gid)
            scene.append(n)
            pts = positions[np.unique(sub[:, :, voff])]
            pts = (M[:3, :3] @ pts.T).T + M[:3, 3]
            if next_label > MAX_LABEL:
                raise ValueError(f"more than {MAX_LABEL} tomatoes; labels no longer fit 8 bits")
            label = next_label
            next_label += 1
            visuals.append((n_name, base, label))
            instances[label] = {
                "plant": plant_dir.name, "node": n_name,
                "centre": [round(float(x), 4) for x in pts.mean(0)],
                "size": [round(float(x), 4) for x in pts.max(0) - pts.min(0)],
            }
        scene.remove(node)
        lib_geom.remove(geom)
    tree.write(dae_out, xml_declaration=True, encoding="utf-8")

    # model.sdf: one visual per node, materials from the original base visuals
    sdf_text = orig_sdf.read_text()
    template = {}
    for m in re.finditer(r'      <visual name="([^"]+)">\n.*?      </visual>\n', sdf_text, re.S):
        template[m.group(1)] = m.group(0)
    model = plant_dir.name
    out = []
    for node_name, base, label in visuals:
        v = template[base]
        v = re.sub(r"\n\s*<!-- semantic label.*?-->", "", v)
        v = re.sub(r"\n\s*<plugin filename=\"gz-sim-label-system\".*?</plugin>", "", v, flags=re.S)
        vname = re.sub(r"[^A-Za-z0-9_]", "_", node_name).lower()
        v = v.replace(f'<visual name="{base}">', f'<visual name="{vname}">', 1)
        v = re.sub(r"<submesh>\s*<name>[^<]*</name>", f"<submesh>\n              <name>{node_name}</name>", v, 1)
        v = v.replace(f"model://{model}/meshes/tomato.dae", f"model://{model}/meshes/tomato_instances.dae")
        # The template's <pbr><metal> blocks don't set metalness, so the SDF default (0.5)
        # applies: half-metal surfaces with no environment map render dark under Ogre2 and
        # barely respond to light (fruit mean 38 -> 54 with metalness 0, 2026-10-11).
        # Plants aren't metal. Ogre1 ignores <pbr>, so this only changes Ogre2 renders.
        if "<metalness>" not in v:
            v = v.replace("</albedo_map>\n", "</albedo_map>\n              <metalness>0.0</metalness>\n")
        if label is not None:
            v = v.replace("      </visual>\n",
                          "        " + LABEL_PLUGIN.format(label=label) + "\n      </visual>\n")
        out.append(v)
    first = min(m.start() for m in re.finditer(r'      <visual name=', sdf_text))
    last = max(m.end() for m in re.finditer(r'      </visual>\n', sdf_text))
    header = ("      <!-- visuals generated by scripts/split_fruit_instances.py from model.sdf.orig:"
              " one per mesh node, one per tomato (labelled) -->\n")
    sdf_path.write_text(sdf_text[:first] + header + "".join(out) + sdf_text[last:])
    n_fruit = sum(1 for v in visuals if v[2] is not None)
    print(f"{model}: {len(visuals)} visuals ({n_fruit} tomatoes)")
    return next_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models_dir")
    ap.add_argument("--only", default="", help="comma-separated plant names (default: all tomato_*)")
    args = ap.parse_args()
    models_dir = Path(args.models_dir)
    plants = sorted(d for d in models_dir.glob("tomato_*") if (d / "meshes" / "tomato.dae").exists())
    if args.only:
        keep = set(args.only.split(","))
        plants = [d for d in plants if d.name in keep]
    if not plants:
        sys.exit(f"no tomato_* models under {models_dir}")
    instances, label = {}, 1
    for d in plants:
        label = process_plant(d, label, instances)
    (models_dir / "fruit_instances.json").write_text(json.dumps(instances, indent=1))
    print(f"{len(instances)} tomatoes labelled 1..{label - 1}; table in {models_dir / 'fruit_instances.json'}")


if __name__ == "__main__":
    main()

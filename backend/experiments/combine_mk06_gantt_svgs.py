from copy import deepcopy
from pathlib import Path
import re
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


BASE_DIR = Path(r"c:\Users\shiver\Desktop\Sched_LLM\backend\data\fjsp_epl_benchmark")
TOP_SVG = BASE_DIR / "mk06_gantt_traditional_spt.svg"
BOTTOM_SVG = BASE_DIR / "mk06_gantt_framework_coop_rh.svg"
OUT_SVG = BASE_DIR / "mk06_gantt_combined.svg"


def strip_unit(value: str) -> float:
    match = re.match(r"([0-9.]+)", value)
    if not match:
        raise ValueError(f"Cannot parse numeric value from: {value}")
    return float(match.group(1))


def prefix_svg_ids(root: ET.Element, prefix: str) -> ET.Element:
    id_map = {}
    for elem in root.iter():
        elem_id = elem.get("id")
        if elem_id:
            new_id = f"{prefix}_{elem_id}"
            id_map[elem_id] = new_id
            elem.set("id", new_id)

    for elem in root.iter():
        for attr_name, attr_value in list(elem.attrib.items()):
            if attr_value.startswith("url(#") and attr_value.endswith(")"):
                old = attr_value[5:-1]
                if old in id_map:
                    elem.set(attr_name, f"url(#{id_map[old]})")
            elif attr_name.endswith("href") and attr_value.startswith("#"):
                old = attr_value[1:]
                if old in id_map:
                    elem.set(attr_name, f"#{id_map[old]}")
    return root


def load_svg(path: Path, prefix: str):
    tree = ET.parse(path)
    root = tree.getroot()
    width = strip_unit(root.get("width", "0"))
    height = strip_unit(root.get("height", "0"))
    prefixed_root = prefix_svg_ids(root, prefix)
    return prefixed_root, width, height


def main():
    top_root, top_w, top_h = load_svg(TOP_SVG, "top")
    bottom_root, bottom_w, bottom_h = load_svg(BOTTOM_SVG, "bottom")

    total_w = max(top_w, bottom_w)
    total_h = top_h + bottom_h

    merged = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "width": f"{total_w}pt",
            "height": f"{total_h}pt",
            "viewBox": f"0 0 {total_w} {total_h}",
        },
    )

    ET.SubElement(
        merged,
        f"{{{SVG_NS}}}rect",
        {
            "x": "0",
            "y": "0",
            "width": str(total_w),
            "height": str(total_h),
            "fill": "white",
        },
    )

    top_group = ET.SubElement(merged, f"{{{SVG_NS}}}g", {"transform": "translate(0,0)"})
    bottom_group = ET.SubElement(merged, f"{{{SVG_NS}}}g", {"transform": f"translate(0,{top_h})"})

    for child in list(top_root):
        top_group.append(deepcopy(child))
    for child in list(bottom_root):
        bottom_group.append(deepcopy(child))

    ET.ElementTree(merged).write(OUT_SVG, encoding="utf-8", xml_declaration=True)
    print(f"saved to: {OUT_SVG}")


if __name__ == "__main__":
    main()

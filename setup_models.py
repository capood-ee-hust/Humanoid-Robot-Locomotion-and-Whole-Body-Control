"""Download the Unitree G1 model from MuJoCo Menagerie into models/unitree_g1/ (about 38 MB)."""
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_g1/{}"
DEST = Path(__file__).resolve().parent / "models" / "unitree_g1"
FILES = ["scene.xml", "g1.xml", "LICENSE", "README.md"]


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "hmp-setup"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())


def mesh_files(xml_path):
    """Mesh files referenced by g1.xml (relative to its assets/ folder)."""
    root = ET.parse(xml_path).getroot()
    return sorted({m.get("file") for m in root.iter("mesh") if m.get("file")})


def main():
    if (DEST / "scene.xml").exists() and (DEST / "assets").exists():
        print(f"G1 already in {DEST}, nothing to do")
        return
    for name in FILES:
        download(RAW.format(name), DEST / name)
        print("  ok", name)
    meshes = mesh_files(DEST / "g1.xml")
    for i, fn in enumerate(meshes, 1):
        download(RAW.format("assets/" + fn), DEST / "assets" / fn)
        print(f"\r  ok assets {i}/{len(meshes)}", end="")
    print(f"\ndone: {DEST}")

    import mujoco
    try:
        m = mujoco.MjModel.from_xml_path(str(DEST / "scene.xml"))
    except Exception as e:
        sys.exit(f"the model does not load: {e}")
    print(f"check: G1 loads (nq={m.nq}, nu={m.nu})")


if __name__ == "__main__":
    main()

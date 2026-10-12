"""Download four pinned SMP encoder weights on a networked PC. Python standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCES = {
    "resnet34": ("smp-hub/resnet34.imagenet", "7a57b34f723329ff020b3f8bc41771163c519d0c"),
    "mobilenet_v2": ("smp-hub/mobilenet_v2.imagenet", "e67aa804e17f7b404b629127eabbd224c4e0690b"),
    "mit_b0": ("smp-hub/mit_b0.imagenet", "9ce53d104d92d75aabb00aae70677aaab67e7c84"),
    "mit_b2": ("smp-hub/mit_b2.imagenet", "868ab6f13871dcf8c3d9f90ee4519403475b65ef"),
}

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="pretrained")
    p.add_argument("--endpoint", default="https://huggingface.co")
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, (repo, revision) in SOURCES.items():
        path = root / f"{name}.safetensors"
        url = f"{a.endpoint.rstrip('/')}/{repo}/resolve/{revision}/model.safetensors"
        previous = manifest.get(name, {})
        if path.exists() and previous.get("revision") == revision and previous.get("sha256") == digest(path):
            print(f"{name}: already downloaded and verified", flush=True)
            continue
        print(f"Downloading {name} from {url}", flush=True)
        temporary = path.with_suffix(".safetensors.part")
        request = urllib.request.Request(url, headers={"User-Agent": "vegann-seg-benchmark"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as f:
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                f.write(block)
                received += len(block)
                print(f"\r{name}: {received / 1048576:.1f} MB" + (f" / {total / 1048576:.1f} MB" if total else ""), end="", flush=True)
        print()
        if total and received != total:
            raise RuntimeError(f"Incomplete download: {name}")
        # Reject HTML/error responses and malformed safetensors headers.
        with temporary.open("rb") as f:
            header_size = int.from_bytes(f.read(8), "little")
            if not 2 <= header_size <= min(received - 8, 16 * 1024 * 1024):
                raise ValueError(f"Invalid safetensors file: {name}")
            header = json.loads(f.read(header_size))
            if not isinstance(header, dict) or not any(k != "__metadata__" for k in header):
                raise ValueError(f"No tensors in downloaded file: {name}")
        temporary.replace(path)
        manifest[name] = {"repo_id": repo, "revision": revision, "sha256": digest(path), "bytes": received}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Upload the entire folder: {root.resolve()}")

if __name__ == "__main__":
    main()

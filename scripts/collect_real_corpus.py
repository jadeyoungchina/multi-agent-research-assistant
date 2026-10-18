"""Fetch the allowlisted, immutable upstream Markdown snapshots for real-v1.

Run explicitly; tests and verification never access the network. The snapshot
manifest records exact bytes, licenses, and a line map for the prefixed context.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "pydantic-v1": ("pydantic/pydantic", "v1.10.17", "b9b8eb76c4f37ed5b33e136aea6d4b80ffa673c8"),
    "pydantic-v2": ("pydantic/pydantic", "v2.8.2", "4978ee235bf5f0fa11159b2dfe46068ef3deba0a"),
    "fastapi": ("fastapi/fastapi", "0.112.2", "d00af00d3f15e9a963e2f1baf8f9b4c8357f6f66"),
}
# Topic families, including every version of a source, belong to exactly one split.
FAMILIES = [
    ("pydantic-coercion", "dev", [("pydantic-v2", "docs/concepts/strict_mode.md"), ("pydantic-v2", "docs/concepts/unions.md")]),
    ("pydantic-fields", "dev", [("pydantic-v2", "docs/concepts/fields.md"), ("pydantic-v2", "docs/concepts/alias.md")]),
    ("fastapi-dependencies", "dev", [("fastapi", "docs/en/docs/tutorial/dependencies/index.md"), ("fastapi", "docs/en/docs/tutorial/dependencies/sub-dependencies.md")]),
    ("fastapi-errors", "dev", [("fastapi", "docs/en/docs/tutorial/handling-errors.md"), ("fastapi", "docs/en/docs/tutorial/response-status-code.md")]),
    ("pydantic-models", "test", [("pydantic-v1", "docs/usage/models.md"), ("pydantic-v2", "docs/concepts/models.md")]),
    ("pydantic-validators", "test", [("pydantic-v1", "docs/usage/validators.md"), ("pydantic-v2", "docs/concepts/validators.md")]),
    ("pydantic-serialization", "test", [("pydantic-v1", "docs/usage/exporting_models.md"), ("pydantic-v2", "docs/concepts/serialization.md")]),
    ("pydantic-dataclasses", "test", [("pydantic-v2", "docs/concepts/dataclasses.md"), ("pydantic-v2", "docs/concepts/type_adapter.md")]),
    ("fastapi-response", "test", [("fastapi", "docs/en/docs/tutorial/response-model.md"), ("fastapi", "docs/en/docs/advanced/response-directly.md")]),
    ("fastapi-background", "test", [("fastapi", "docs/en/docs/tutorial/background-tasks.md"), ("fastapi", "docs/en/docs/advanced/events.md")]),
    ("fastapi-security", "test", [("fastapi", "docs/en/docs/tutorial/security/first-steps.md"), ("fastapi", "docs/en/docs/tutorial/security/simple-oauth2.md")]),
    ("fastapi-middleware", "test", [("fastapi", "docs/en/docs/tutorial/middleware.md"), ("fastapi", "docs/en/docs/tutorial/cors.md")]),
]


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "real-v1-corpus-snapshot/1.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        content = response.read(1_000_001)
    if len(content) > 1_000_000:
        raise ValueError("unexpected upstream document size")
    content.decode("utf-8")
    return content


def collect(destination: Path) -> dict:
    retrieved_at = datetime.now(timezone.utc).isoformat()
    entries = []
    for family, split, files in FAMILIES:
        for source_key, upstream_path in files:
            repository, tag, commit = SOURCES[source_key]
            identifier = source_key + "--" + upstream_path.removeprefix("docs/").replace("/", "--")
            entries.append(dict(id=identifier.removesuffix(".md"), family=family, split=split,
                                repository=repository, tag=tag, commit=commit, upstream_path=upstream_path))
    urls = [f"https://raw.githubusercontent.com/{row['repository']}/{row['commit']}/{row['upstream_path']}" for row in entries]
    license_keys = list(SOURCES)
    urls.extend(f"https://raw.githubusercontent.com/{SOURCES[key][0]}/{SOURCES[key][2]}/LICENSE" for key in license_keys)
    # All downloads must succeed before any existing snapshot is replaced.
    with ThreadPoolExecutor(max_workers=4) as pool:
        blobs = list(pool.map(fetch, urls))
    licenses = []
    for key, content, url in zip(license_keys, blobs[len(entries):], urls[len(entries):]):
        if b"Permission is hereby granted, free of charge" not in content:
            raise ValueError("license requires manual inspection")
        filename = f"licenses/{key}-LICENSE.txt"
        path = destination / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        licenses.append(dict(id=key, spdx="MIT", path=filename, sha256=sha256(content).hexdigest(), url=url))
    for row, content, raw_url in zip(entries, blobs, urls):
        raw_path = f"raw/{row['id']}.md"
        corpus_path = f"corpus/{row['id']}.md"
        source_url = f"https://github.com/{row['repository']}/blob/{row['commit']}/{row['upstream_path']}"
        prefix = (f"Snapshot repository: {row['repository']}\nSnapshot version: {row['tag']}\n"
                  f"Snapshot source: {source_url}\n\n---\n\n")
        normalized = prefix + content.decode("utf-8").replace("\r\n", "\n")
        for name, data in ((raw_path, content), (corpus_path, normalized.encode("utf-8"))):
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        row.update(raw_path=raw_path, corpus_path=corpus_path, source_url=source_url, raw_url=raw_url,
                   retrieved_at_utc=retrieved_at, raw_sha256=sha256(content).hexdigest(),
                   corpus_sha256=sha256(normalized.encode("utf-8")).hexdigest(),
                   license_id=next(key for key, value in SOURCES.items() if value[0] == row['repository'] and value[2] == row['commit']),
                   original_line_count=len(content.decode("utf-8").splitlines()),
                   normalized_line_offset=len(prefix.splitlines()))
    manifest = dict(schema_version="real-corpus-v1", dataset_kind="public_documents_ai_authored_questions",
                    normalization="UTF-8; CRLF to LF; six provenance lines prefixed; no include expansion or AI rewriting",
                    split_unit="topic_family_with_all_source_versions", licenses=licenses, documents=entries)
    (destination / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "benchmarks/real-v1")
    args = parser.parse_args()
    manifest = collect(args.destination)
    print(f"Saved {len(manifest['documents'])} pinned documents and {len(manifest['licenses'])} licenses.")

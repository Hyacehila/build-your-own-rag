import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pymupdf
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
)

TOOLS = Path(__file__).resolve().parents[1] / "scripts" / "document_tools.py"


class DocumentToolsTest(unittest.TestCase):
    def test_profile_diagnostics_and_reload_without_project_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "sample.pdf"
            with pymupdf.open() as pdf:
                pdf.new_page(width=600, height=800)
                pdf.save(pdf_path)
            document = DoclingDocument(name="sample")
            document.add_page(page_no=1, size=Size(width=600, height=800))
            provenance = ProvenanceItem(page_no=1, charspan=(0, 7),
                                        bbox=BoundingBox(l=10, t=10, r=100, b=30,
                                                         coord_origin=CoordOrigin.TOPLEFT))
            heading = document.add_heading(text="Heading", level=1, prov=provenance)
            document.add_text(label=DocItemLabel.TEXT, text="Example", parent=heading, prov=provenance)
            (root / "raw.json").write_text(document.model_dump_json(), encoding="utf-8")
            profile = {"excluded_pages": [], "boundaries": [1], "heading_levels": {heading.self_ref: 1}}
            (root / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
            def run(*args):
                subprocess.run([sys.executable, str(TOOLS), *map(str, args)], check=True, cwd=root,
                               capture_output=True, text=True)
            run("hierarchize", "--document", root / "raw.json", "--pdf", pdf_path,
                "--output", root / "hierarchized.json")
            run("apply-profile", "--document", root / "raw.json", "--pdf", pdf_path,
                "--profile", root / "profile.json", "--output", root / "clean.json",
                "--decisions", root / "decisions.json")
            run("check", "--document", root / "clean.json", "--pdf", pdf_path)
            run("diagnose", "--document", root / "clean.json", "--pdf", pdf_path,
                "--output", root / "quality.json")
            run("review-packet", "--document", root / "clean.json", "--pdf", pdf_path,
                "--output", root / "packet.json")
            packet = json.loads((root / "packet.json").read_text(encoding="utf-8"))
            self.assertFalse(packet["review_complete"])
            review = {"doc_id": "sample", "reviewer": "test", "final_tree_reviewed": True,
                      "reviewed_heading_refs": [item["ref"] for item in packet["headings"]],
                      "reviewed_headings": packet["headings"],
                      "confirmed_wrong_parent_remaining": [], "representative_pages_viewed": [1]}
            (root / "review.json").write_text(json.dumps(review), encoding="utf-8")
            run("manifest", "--entry", "sample", pdf_path, root / "clean.json", root / "review.json",
                "--output", root / "manifest.json")
            self.assertEqual(json.loads((root / "manifest.json").read_text(encoding="utf-8"))["documents"][0]["pdf"],
                             "sample.pdf")
            cleaned = DoclingDocument.model_validate_json((root / "clean.json").read_text(encoding="utf-8"))
            self.assertEqual([x.text for x in cleaned.texts], ["Heading", "Example"])
            self.assertTrue(json.loads((root / "decisions.json").read_text(encoding="utf-8"))["content_unchanged"])

    def test_mineru_candidate_retains_explicit_physical_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = {"page_size": [600, 800], "source_page_index": 2,
                       "uploaded_page_index": 0, "doc_id": "sample"}
            items = [{"type": "table", "page_idx": 0, "bbox": [100, 100, 900, 800],
                      "table_body": "<table><tr><th>Year</th><th>Value</th></tr><tr><td>2024</td><td>42</td></tr></table>"}]
            (root / "map.json").write_text(json.dumps(mapping), encoding="utf-8")
            (root / "items.json").write_text(json.dumps(items), encoding="utf-8")
            subprocess.run([sys.executable, str(TOOLS), "mineru-convert", "--content-list", str(root / "items.json"),
                            "--page-map", str(root / "map.json"), "--output", str(root / "candidate.json"),
                            "--report", str(root / "report.json")], check=True, cwd=root, capture_output=True)
            candidate = DoclingDocument.model_validate_json((root / "candidate.json").read_text(encoding="utf-8"))
            self.assertEqual(candidate.tables[0].prov[0].page_no, 3)
            self.assertEqual(candidate.tables[0].data.table_cells[-1].text, "42")

    def test_mineru_layout_conversion_preserves_source_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "sample.pdf"
            with pymupdf.open() as source:
                source.new_page(width=600, height=800)
                source.new_page(width=600, height=800)
                source.save(pdf)
            page = {"uploaded_page_index": 0, "page_number": 2, "page_size": [600, 800], "owner": True}
            manifest = {"files": [{"data_id": "part", "pages": [page]}],
                        "excluded_page_numbers": [], "unique_body_pages": 1}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            body = {"type": "table_body", "bbox": [10, 30, 180, 80], "lines": [{"spans": [
                {"html": "<table><tr><th>Item</th><th>Value</th></tr><tr><td>A</td><td>42</td></tr></table>"}
            ]}]}
            layout = {"_version_name": "3.4.4", "pdf_info": [{"page_idx": 0, "page_size": [600, 800],
                      "preproc_blocks": [{"type": "table", "bbox": [10, 30, 180, 80], "blocks": [body]}],
                      "discarded_blocks": []}]}
            archive = root / "result.zip"
            with zipfile.ZipFile(archive, "w") as result:
                result.writestr("layout.json", json.dumps(layout))
            results = {"complete": True, "results": [{"data_id": "part", "state": "done", "zip_file": str(archive)}]}
            (root / "results.json").write_text(json.dumps(results), encoding="utf-8")
            subprocess.run([sys.executable, str(TOOLS), "mineru-layout", "--bundle", str(root), "--pdf", str(pdf),
                            "--doc-id", "sample", "--output", str(root / "output")],
                           check=True, cwd=root, capture_output=True)
            candidate = DoclingDocument.model_validate_json((root / "output/document.json").read_text(encoding="utf-8"))
            self.assertEqual(candidate.tables[0].prov[0].page_no, 2)
            self.assertEqual(candidate.tables[0].data.table_cells[-1].text, "42")


if __name__ == "__main__":
    unittest.main()

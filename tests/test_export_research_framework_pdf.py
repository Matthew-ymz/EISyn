import tempfile
import unittest
from pathlib import Path

from scripts.export_research_framework_pdf import (
    collect_svg_image_references,
    rewrite_markdown_for_pdf_export,
)


class ExportResearchFrameworkPdfTests(unittest.TestCase):
    def test_rewrite_only_updates_image_references(self) -> None:
        source = """
Inline mention `single_node_unique_budget.svg` should stay unchanged.

<img src="../fig/a.svg" alt="A" width="40%" />

![Diagram](../fig/b.svg)

Regular link [manifest](../fig/figure_manifest.svg) should stay unchanged.
"""

        rewritten = rewrite_markdown_for_pdf_export(source)

        self.assertIn("`single_node_unique_budget.svg`", rewritten)
        self.assertIn('<img src="../fig/a.png" alt="A" width="40%" />', rewritten)
        self.assertIn("![Diagram](../fig/b.png)", rewritten)
        self.assertIn("[manifest](../fig/figure_manifest.svg)", rewritten)

    def test_rewrite_converts_centered_html_image_block_to_markdown_images(self) -> None:
        source = """
<p align="center">
  <img src="../fig/a.svg" alt="A" width="48%" />
  <img src="../fig/b.svg" alt="B" width="48%" />
</p>
"""

        rewritten = rewrite_markdown_for_pdf_export(source)

        self.assertNotIn("<p align=", rewritten)
        self.assertNotIn("<img", rewritten)
        self.assertIn("![A](../fig/a.png){ width=48% }", rewritten)
        self.assertIn("![B](../fig/b.png){ width=48% }", rewritten)

    def test_collect_svg_image_references_finds_html_and_markdown_images(self) -> None:
        source = """
<img src="../fig/a.svg" alt="A" width="40%" />
![Diagram](../fig/b.svg)
[Regular link](../fig/c.svg)
"""

        references = collect_svg_image_references(source)

        self.assertEqual(references, ["../fig/a.svg", "../fig/b.svg"])

    def test_rewrite_does_not_modify_source_file_in_place(self) -> None:
        source = '# Demo\n![Figure](../fig/demo.svg)\n'
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "研究框架.md"
            source_path.write_text(source, encoding="utf-8")

            original = source_path.read_text(encoding="utf-8")
            rewritten = rewrite_markdown_for_pdf_export(original)

            self.assertEqual(source_path.read_text(encoding="utf-8"), original)
            self.assertNotEqual(rewritten, original)
            self.assertIn("../fig/demo.png", rewritten)


if __name__ == "__main__":
    unittest.main()

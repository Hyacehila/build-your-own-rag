"""Repeat all declared leading table headers without modifying the fact layer."""

from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer


class MultirowTableSerializer(MarkdownTableSerializer):
    def serialize(self, *, item, doc_serializer, doc, **kwargs):
        result = super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)
        if kwargs.get("_nested_in_table"):
            return result
        depth = 1
        for row in item.data.grid[1:]:
            if not any(cell.column_header for cell in row):
                break
            depth += 1
        # A whole table marked as header is ambiguous. Keep the original rows;
        # collapsing them would leave HybridChunker with no body to segment.
        if depth <= 1 or depth >= item.data.num_rows:
            return result
        lines = result.text.splitlines()
        separator = next((i for i, line in enumerate(lines) if self._SEPARATOR_ROW_RE.match(line)), None)
        if separator is None or separator == 0:
            return result
        header_lines = [lines[separator - 1], *lines[separator + 1 : separator + depth]]
        # The upstream serializer already escapes pipes/newlines and preserves
        # numeric strings. Collapse only declared leading header rows, column by
        # column, so HybridChunker's repeated prefix includes years and units.
        rows = [[cell.strip() for cell in line.split("|")[1:-1]] for line in header_lines]
        if len(rows) != depth or any(len(row) != item.data.num_cols for row in rows):
            return result
        header = (
            "| "
            + " | ".join(" / ".join(dict.fromkeys(cell for cell in col if cell)) for col in zip(*rows))
            + " |"
        )
        text = "\n".join([*lines[: separator - 1], header, lines[separator], *lines[separator + depth :]])
        return create_ser_result(text=text, span_source=[result])

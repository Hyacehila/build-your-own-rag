from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument, ImageRef, Size
from document_tools import embed_images
from PIL import Image


def test_relative_image_is_embedded_and_survives_source_removal(tmp_path):
    directory = tmp_path / "assets"
    directory.mkdir()
    image_path = directory / "picture.png"
    Image.new("RGB", (2, 2), "red").save(image_path)
    document = DoclingDocument(name="sample")
    picture = document.add_picture()
    picture.image = ImageRef(mimetype="image/png", dpi=72, size=Size(width=2, height=2),
                             uri=Path("assets/picture.png"))
    embed_images(document, tmp_path)
    image_path.unlink()
    assert str(document.pictures[0].image.uri).startswith("data:image/png;base64,")
    assert document.pictures[0].image.pil_image.size == (2, 2)


def test_missing_image_is_rejected(tmp_path):
    document = DoclingDocument(name="sample")
    picture = document.add_picture()
    picture.image = ImageRef(mimetype="image/png", dpi=72, size=Size(width=2, height=2),
                             uri=Path("assets/missing.png"))
    with pytest.raises(FileNotFoundError):
        embed_images(document, tmp_path)

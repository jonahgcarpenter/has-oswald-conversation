"""Verify HA can serve every brand variant from the single bundled icon."""

from pathlib import Path

import pytest
from homeassistant.components.brands import _read_brand_file
from homeassistant.components.brands.const import ALLOWED_IMAGES

BRAND_DIR = (
    Path(__file__).resolve().parents[1] / "custom_components/oswald_conversation/brand"
)


@pytest.mark.parametrize("image", sorted(ALLOWED_IMAGES))
def test_brand_image_fallback(image):
    assert _read_brand_file(BRAND_DIR, image) == (BRAND_DIR / "icon.png").read_bytes()

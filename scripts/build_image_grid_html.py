from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
FILENAME_PATTERN = re.compile(r"^(?P<name>.+)_(?P<row>\d+)_(?P<col>\d+)$")


@dataclass(frozen=True)
class ImageTile:
    path: Path
    base_name: str
    row: int
    col: int
    data_uri: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an HTML visualization for folders with images named as "
            "<image_name>_<row_idx>_<col_idx>.<ext>."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Root directory with structure ./<cell_name>/<image_name>_<row>_<col>.jpg",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("grid_visualization.html"),
        help="Output HTML file path (default: ./grid_visualization.html).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Image Grid Visualization",
        help="Page title for generated HTML.",
    )
    return parser.parse_args()


def to_data_uri(image_path: Path) -> str:
    raw = image_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    mime, _ = mimetypes.guess_type(image_path.name)
    if mime is None:
        mime = "application/octet-stream"
    return f"data:{mime};base64,{encoded}"


def parse_tile(image_path: Path) -> ImageTile | None:
    stem = image_path.stem
    match = FILENAME_PATTERN.match(stem)
    if not match:
        return None

    row = int(match.group("row"))
    col = int(match.group("col"))
    base_name = match.group("name")
    return ImageTile(
        path=image_path,
        base_name=base_name,
        row=row,
        col=col,
        data_uri=to_data_uri(image_path),
    )


def collect_cell_tiles(cell_dir: Path) -> list[ImageTile]:
    tiles: list[ImageTile] = []
    for file_path in sorted(cell_dir.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        tile = parse_tile(file_path)
        if tile is not None:
            tiles.append(tile)
    return tiles


def build_cell_block(cell_name: str, tiles: list[ImageTile]) -> str:
    if not tiles:
        return f"""
        <section class="cell-block">
          <h2>{html.escape(cell_name)}</h2>
          <p class="empty-note">No valid image files found for this cell.</p>
        </section>
        """

    max_col = max(tile.col for tile in tiles)

    tile_html = []
    for tile in sorted(tiles, key=lambda t: (t.row, t.col, t.base_name, t.path.name)):
        tile_html.append(
            f"""
            <figure class="tile" style="grid-row: {tile.row + 1}; grid-column: {tile.col + 1};">
              <img src="{tile.data_uri}" alt="{html.escape(tile.path.name)}" loading="lazy" />
              <figcaption>{html.escape(tile.path.name)}</figcaption>
            </figure>
            """
        )

    return f"""
    <section class="cell-block">
      <h2>{html.escape(cell_name)}</h2>
      <div class="grid" style="grid-template-columns: repeat({max_col + 1}, minmax(160px, 1fr));">
        {''.join(tile_html)}
      </div>
    </section>
    """


def build_html(title: str, cell_blocks: list[str]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light dark;
    }}

    body {{
      margin: 24px;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      line-height: 1.4;
      background: Canvas;
      color: CanvasText;
    }}

    h1 {{
      margin-top: 0;
      margin-bottom: 16px;
      font-size: 1.6rem;
    }}

    .meta {{
      margin-bottom: 24px;
      opacity: 0.8;
      font-size: 0.95rem;
    }}

    .cell-block {{
      border: 1px solid color-mix(in srgb, CanvasText 25%, transparent);
      border-radius: 10px;
      padding: 14px;
      margin-bottom: 18px;
      background: color-mix(in srgb, Canvas 90%, CanvasText 3%);
    }}

    .cell-block > h2 {{
      margin: 0 0 12px 0;
      font-size: 1.1rem;
    }}

    .grid {{
      display: grid;
      gap: 12px;
      align-items: start;
    }}

    .tile {{
      margin: 0;
      border: 1px solid color-mix(in srgb, CanvasText 20%, transparent);
      border-radius: 8px;
      overflow: hidden;
      background: color-mix(in srgb, Canvas 92%, CanvasText 6%);
    }}

    .tile img {{
      width: 100%;
      height: auto;
      display: block;
      object-fit: contain;
      background: #00000010;
    }}

    .tile figcaption {{
      padding: 6px 8px;
      font-size: 0.82rem;
      opacity: 0.9;
      word-break: break-word;
    }}

    .empty-note {{
      margin: 0;
      opacity: 0.75;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="meta">Each subfolder is rendered as one block. Image positions are derived from filename suffix <code>_row_col</code>.</p>
  {''.join(cell_blocks)}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output = args.output

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    cell_dirs = sorted([path for path in input_dir.iterdir() if path.is_dir()])
    cell_blocks: list[str] = []

    for cell_dir in cell_dirs:
        tiles = collect_cell_tiles(cell_dir)
        cell_blocks.append(build_cell_block(cell_dir.name, tiles))

    html_doc = build_html(args.title, cell_blocks)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")
    print(f"HTML visualization saved to: {output.resolve()}")


if __name__ == "__main__":
    main()

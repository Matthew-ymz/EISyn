"""Compose related brain-report figures into compact appendix plates."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
WHITE = (255, 255, 255)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _open(relative_path: str) -> Image.Image:
    return Image.open(ROOT / relative_path).convert("RGB")


def _fit_width(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _fit_height(image: Image.Image, height: int) -> Image.Image:
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def compose_grid(
    sources: list[str],
    destination: str,
    labels: list[str],
    columns: int = 2,
    panel_width: int = 4200,
    gap: int = 80,
    label_band: int = 110,
) -> None:
    panels = [_fit_width(_open(path), panel_width) for path in sources]
    rows = (len(panels) + columns - 1) // columns
    row_heights = [
        max(panel.height for panel in panels[row * columns : (row + 1) * columns])
        for row in range(rows)
    ]
    canvas_width = columns * panel_width + (columns - 1) * gap
    canvas_height = sum(row_heights) + rows * label_band + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)
    draw = ImageDraw.Draw(canvas)
    font = _font(72)

    y = 0
    for row in range(rows):
        row_height = row_heights[row]
        for column in range(columns):
            index = row * columns + column
            if index >= len(panels):
                break
            panel = panels[index]
            x = column * (panel_width + gap)
            draw.text((x + 18, y + 10), labels[index], fill="black", font=font)
            panel_y = y + label_band + (row_height - panel.height) // 2
            canvas.paste(panel, (x, panel_y))
        y += label_band + row_height + gap

    output = ROOT / destination
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)


def compose_row(
    sources: list[str],
    destination: str,
    labels: list[str],
    panel_height: int = 2554,
    gap: int = 80,
    label_band: int = 100,
) -> None:
    panels = [_fit_height(_open(path), panel_height) for path in sources]
    canvas_width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (canvas_width, panel_height + label_band), WHITE)
    draw = ImageDraw.Draw(canvas)
    font = _font(68)

    x = 0
    for panel, label in zip(panels, labels, strict=True):
        draw.text((x + 18, 8), label, fill="black", font=font)
        canvas.paste(panel, (x, label_band))
        x += panel.width + gap

    output = ROOT / destination
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)


def main() -> None:
    robustness = "results/hcp_schaefer500_phi_hyperparameter_robustness"
    compose_grid(
        [
            f"{robustness}/hyperparameter_robustness_overview.png",
            f"{robustness}/hyperparameter_task_margins.png",
            f"{robustness}/prediction_error_overview.png",
            f"{robustness}/prediction_error_by_condition.png",
        ],
        "fig/brain_hcp500_robustness_plate.png",
        ["A", "B", "C", "D"],
    )

    for scale in (500, 1000):
        prefix = f"results/hcp_schaefer{scale}"
        compose_row(
            [
                f"{prefix}_yeo7_pc1_phi_null_all/observed_minus_null.png",
                f"{prefix}_yeo7_module_phi_decomposition/top_core_consistency.png",
            ],
            f"fig/brain_hcp{scale}_yeo7_phi_null_summary.png",
            ["A", "B"],
        )


if __name__ == "__main__":
    main()

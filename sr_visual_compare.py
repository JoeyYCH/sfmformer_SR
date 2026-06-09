"""
SR 論文風格的視覺化比較圖產生器
功能：
1. 在原圖上標示 ROI (綠框)
2. 將 ROI 區域放大後貼在圖片角落 (紅框)
3. 多張圖片橫向拼接，加上標題文字
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys


def create_sr_comparison(
    image_paths,          # list of image paths (LR, Method1, Method2, ...)
    labels,               # list of labels  ("LR (×4)", "ESRGAN", "Ours", ...)
    roi_box,              # (x1, y1, x2, y2) 感興趣區域座標 (基於 HR 圖片尺寸)
    output_path,          # 輸出檔案路徑
    zoom_scale=3,         # ROI 放大倍率
    zoom_position='br',   # 放大區域位置: 'br'=右下, 'bl'=左下, 'tr'=右上, 'tl'=左上
    roi_color=(0, 255, 0),        # ROI 框顏色 (綠色)
    zoom_border_color=(255, 0, 0), # 放大框顏色 (紅色)
    border_width=2,       # 框線粗細
    label_font_size=32,   # 標題文字大小
    padding=10,           # 圖片之間的間距
    zoom_margin=8,        # 放大框與圖片邊緣的距離
    bg_color=(255, 255, 255),  # 背景色
):
    """
    生成 SR 論文風格的視覺化比較圖

    Parameters
    ----------
    image_paths : list[str]
        各方法產出的 SR 圖片路徑列表
    labels : list[str]
        對應每張圖片的標題文字
    roi_box : tuple (x1, y1, x2, y2)
        在圖片上要放大的感興趣區域座標
    output_path : str
        輸出拼接圖的檔案路徑
    zoom_scale : int
        ROI 裁切後的放大倍率
    zoom_position : str
        放大區域要貼在圖片的哪個角落
        'br'=右下, 'bl'=左下, 'tr'=右上, 'tl'=左上
    """

    images = []
    for p in image_paths:
        img = Image.open(p).convert('RGB')
        images.append(img)

    # 確保所有圖片尺寸一致
    base_w, base_h = images[0].size
    for i, img in enumerate(images):
        if img.size != (base_w, base_h):
            print(f"[Warning] {labels[i]} size {img.size} != base {(base_w, base_h)}, resizing...")
            images[i] = img.resize((base_w, base_h), Image.LANCZOS)

    x1, y1, x2, y2 = roi_box
    roi_w = x2 - x1
    roi_h = y2 - y1

    # 放大後的尺寸
    zoomed_w = roi_w * zoom_scale
    zoomed_h = roi_h * zoom_scale

    # 處理每張圖片
    annotated_images = []
    for img in images:
        canvas = img.copy()
        draw = ImageDraw.Draw(canvas)

        # 1. 畫 ROI 綠框
        for offset in range(border_width):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=roi_color
            )

        # 2. 裁切 ROI 並放大
        roi_crop = img.crop((x1, y1, x2, y2))
        roi_zoomed = roi_crop.resize((zoomed_w, zoomed_h), Image.NEAREST)
        # ↑ 用 NEAREST 保持像素銳利度，讓差異更明顯
        # 如果你希望平滑一點，可以改成 Image.LANCZOS

        # 3. 計算放大框的位置
        if zoom_position == 'br':
            zx = base_w - zoomed_w - zoom_margin
            zy = base_h - zoomed_h - zoom_margin
        elif zoom_position == 'bl':
            zx = zoom_margin
            zy = base_h - zoomed_h - zoom_margin
        elif zoom_position == 'tr':
            zx = base_w - zoomed_w - zoom_margin
            zy = zoom_margin
        elif zoom_position == 'tl':
            zx = zoom_margin
            zy = zoom_margin
        else:
            zx = base_w - zoomed_w - zoom_margin
            zy = base_h - zoomed_h - zoom_margin

        # 4. 貼上放大後的 ROI
        canvas.paste(roi_zoomed, (zx, zy))

        # 5. 畫紅框
        for offset in range(border_width):
            draw.rectangle(
                [zx - offset, zy - offset,
                 zx + zoomed_w + offset, zy + zoomed_h + offset],
                outline=zoom_border_color
            )

        annotated_images.append(canvas)

    # ==================== 拼接所有圖片 ====================
    n = len(annotated_images)
    label_area_height = label_font_size + 16  # 文字區域高度

    total_w = n * base_w + (n - 1) * padding
    total_h = base_h + label_area_height

    result = Image.new('RGB', (total_w, total_h), bg_color)
    draw_result = ImageDraw.Draw(result)

    # 嘗試載入字型
    font = None
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, label_font_size)
                break
            except:
                continue
    if font is None:
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(annotated_images, labels)):
        x_offset = i * (base_w + padding)

        # 貼圖
        result.paste(img, (x_offset, 0))

        # 文字置中
        bbox = draw_result.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        tx = x_offset + (base_w - tw) // 2
        ty = base_h + 4

        draw_result.text((tx, ty), label, fill=(0, 0, 0), font=font)

    result.save(output_path, quality=95)
    print(f"[Done] Saved to: {output_path}")
    print(f"       Size: {total_w} × {total_h}")
    return output_path


def create_sr_comparison_multi_roi(
    image_paths,
    labels,
    roi_boxes,             # list of (x1, y1, x2, y2)
    output_path,
    zoom_scale=3,
    zoom_positions=None,   # list of positions, e.g. ['tl', 'br']
    roi_colors=None,       # list of colors for each ROI
    zoom_border_colors=None,
    border_width=2,
    label_font_size=32,
    padding=10,
    zoom_margin=8,
    bg_color=(255, 255, 255),
):
    """支援多個 ROI 區域的版本"""

    if zoom_positions is None:
        # 預設位置：根據 ROI 數量自動分配角落
        default_pos = ['br', 'tl', 'tr', 'bl']
        zoom_positions = [default_pos[i % 4] for i in range(len(roi_boxes))]

    if roi_colors is None:
        color_palette = [(0, 255, 0), (0, 200, 255), (255, 255, 0), (255, 128, 0)]
        roi_colors = [color_palette[i % len(color_palette)] for i in range(len(roi_boxes))]

    if zoom_border_colors is None:
        zoom_border_colors = [(255, 0, 0)] * len(roi_boxes)

    images = [Image.open(p).convert('RGB') for p in image_paths]
    base_w, base_h = images[0].size
    for i, img in enumerate(images):
        if img.size != (base_w, base_h):
            images[i] = img.resize((base_w, base_h), Image.LANCZOS)

    annotated_images = []
    for img in images:
        canvas = img.copy()
        draw = ImageDraw.Draw(canvas)

        for roi_idx, (roi_box, pos, rc, zbc) in enumerate(
            zip(roi_boxes, zoom_positions, roi_colors, zoom_border_colors)
        ):
            x1, y1, x2, y2 = roi_box
            roi_w, roi_h = x2 - x1, y2 - y1
            zoomed_w, zoomed_h = roi_w * zoom_scale, roi_h * zoom_scale

            # 綠框
            for offset in range(border_width):
                draw.rectangle(
                    [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                    outline=rc
                )

            # 裁切放大
            roi_crop = img.crop((x1, y1, x2, y2))
            roi_zoomed = roi_crop.resize((zoomed_w, zoomed_h), Image.NEAREST)

            # 位置
            if pos == 'br':
                zx = base_w - zoomed_w - zoom_margin
                zy = base_h - zoomed_h - zoom_margin
            elif pos == 'bl':
                zx = zoom_margin
                zy = base_h - zoomed_h - zoom_margin
            elif pos == 'tr':
                zx = base_w - zoomed_w - zoom_margin
                zy = zoom_margin
            elif pos == 'tl':
                zx = zoom_margin
                zy = zoom_margin
            else:
                zx = base_w - zoomed_w - zoom_margin
                zy = base_h - zoomed_h - zoom_margin

            canvas.paste(roi_zoomed, (zx, zy))

            for offset in range(border_width):
                draw.rectangle(
                    [zx - offset, zy - offset,
                     zx + zoomed_w + offset, zy + zoomed_h + offset],
                    outline=zbc
                )

        annotated_images.append(canvas)

    # 拼接
    n = len(annotated_images)
    label_area_height = label_font_size + 16
    total_w = n * base_w + (n - 1) * padding
    total_h = base_h + label_area_height

    result = Image.new('RGB', (total_w, total_h), bg_color)
    draw_result = ImageDraw.Draw(result)

    font = None
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, label_font_size)
                break
            except:
                continue
    if font is None:
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(annotated_images, labels)):
        x_offset = i * (base_w + padding)
        result.paste(img, (x_offset, 0))
        bbox = draw_result.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        tx = x_offset + (base_w - tw) // 2
        ty = base_h + 4
        draw_result.text((tx, ty), label, fill=(0, 0, 0), font=font)

    result.save(output_path, quality=95)
    print(f"[Done] Saved to: {output_path}")
    return output_path


# ==================== Demo ====================
if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════╗
║  SR Paper Visual Comparison Tool                     ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  使用方式 (單一 ROI):                                  ║
║                                                      ║
║    from sr_visual_compare import *                   ║
║                                                      ║
║    create_sr_comparison(                             ║
║        image_paths=[                                 ║
║            'results/img001_LR.png',                  ║
║            'results/img001_SwinIR.png',              ║
║            'results/img001_PFT.png',                 ║
║            'results/img001_Ours.png',                ║
║        ],                                            ║
║        labels=[                                      ║
║            'LR (×4)', 'SwinIR', 'PFT', 'Ours'       ║
║        ],                                            ║
║        roi_box=(120, 80, 180, 140),                  ║
║        output_path='comparison.png',                 ║
║        zoom_scale=4,                                 ║
║        zoom_position='br',                           ║
║    )                                                 ║
║                                                      ║
║  使用方式 (多 ROI):                                    ║
║                                                      ║
║    create_sr_comparison_multi_roi(                   ║
║        image_paths=[...],                            ║
║        labels=[...],                                 ║
║        roi_boxes=[                                   ║
║            (30, 20, 80, 70),    # 綠框               ║
║            (200, 150, 280, 230) # 青框               ║
║        ],                                            ║
║        zoom_positions=['tl', 'br'],                  ║
║        output_path='comparison_2roi.png',            ║
║        zoom_scale=3,                                 ║
║    )                                                 ║
║                                                      ║
║  小技巧:                                              ║
║  - roi_box 座標可以用圖片瀏覽器量測                      ║
║  - zoom_scale=3~5 視 ROI 大小調整                     ║
║  - 用 NEAREST 插值讓像素差異更明顯                      ║
║  - 如需平滑放大，修改 Image.NEAREST → Image.LANCZOS    ║
╚══════════════════════════════════════════════════════╝
""")

    # 生成假的範例圖片做 demo
    print("Generating demo images...")
    os.makedirs('/home/claude/demo_sr', exist_ok=True)

    for i, name in enumerate(['LR', 'SwinIR', 'PFT', 'Ours']):
        np.random.seed(42 + i)
        # 模擬不同品質的 SR 結果
        base = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        if i == 0:  # LR - 模糊
            from PIL import ImageFilter
            img = Image.fromarray(base).filter(ImageFilter.GaussianBlur(radius=3))
        elif i == 3:  # Ours - 最清晰
            img = Image.fromarray(base)
        else:
            img = Image.fromarray(base).filter(ImageFilter.GaussianBlur(radius=1))
        img.save(f'/home/claude/demo_sr/{name}.png')

    # 單 ROI demo
    create_sr_comparison(
        image_paths=[f'/home/claude/demo_sr/{n}.png' for n in ['LR', 'SwinIR', 'PFT', 'Ours']],
        labels=['LR (×4)', 'SwinIR', 'PFT', 'Ours'],
        roi_box=(30, 40, 90, 100),
        output_path='/home/claude/demo_sr/comparison_demo.png',
        zoom_scale=3,
        zoom_position='br',
        border_width=2,
    )

    # 多 ROI demo
    create_sr_comparison_multi_roi(
        image_paths=[f'/home/claude/demo_sr/{n}.png' for n in ['LR', 'SwinIR', 'PFT', 'Ours']],
        labels=['LR (×4)', 'SwinIR', 'PFT', 'Ours'],
        roi_boxes=[
            (20, 20, 70, 70),
            (150, 150, 220, 220),
        ],
        zoom_positions=['tl', 'br'],
        output_path='/home/claude/demo_sr/comparison_multi_roi_demo.png',
        zoom_scale=3,
    )

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("/root/autodl-tmp/ostrakon_task0/images")
OUT.mkdir(parents=True, exist_ok=True)

try:
    title_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
    )
    text_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
    )
except OSError:
    title_font = ImageFont.load_default()
    text_font = ImageFont.load_default()


def canvas(title):
    image = Image.new("RGB", (768, 512), "#f4f1e8")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), title, fill="black", font=title_font)
    return image, draw


def shelves(draw):
    for y in (155, 305, 455):
        draw.rectangle((25, y, 743, y + 12), fill="#555555")


image, draw = canvas("SCENE 1 - SHELF INVENTORY")
shelves(draw)
colors = ["#d94b4b", "#3977c3", "#3b9b58", "#e0a72f"]
for row in range(3):
    for col in range(7):
        x, y = 45 + col * 99, 65 + row * 150
        draw.rectangle(
            (x, y, x + 70, y + 82),
            fill=colors[(row + col) % 4],
            outline="black",
            width=2,
        )
        draw.text((x + 19, y + 26), f"P{row * 7 + col + 1}", fill="white", font=text_font)
image.save(OUT / "01_inventory.jpg", quality=95)

image, draw = canvas("SCENE 2 - SAFETY COMPLIANCE")
draw.rectangle((500, 70, 720, 445), fill="#2e8b57", outline="black", width=4)
draw.text((535, 105), "FIRE", fill="white", font=title_font)
draw.text((535, 145), "EXIT", fill="white", font=title_font)
draw.polygon([(610, 220), (660, 250), (610, 280)], fill="white")
for box in [(455, 320, 570, 445), (565, 345, 680, 445), (400, 375, 500, 445)]:
    draw.rectangle(box, fill="#a66a3f", outline="black", width=3)
    draw.line((box[0], box[1], box[2], box[3]), fill="#704020", width=2)
draw.text(
    (25, 420),
    "Cardboard boxes obstruct the emergency exit",
    fill="#b00020",
    font=text_font,
)
image.save(OUT / "02_compliance.jpg", quality=95)

image, draw = canvas("SCENE 3 - PRICE LABEL OCR")
labels = [("MILK 1L", "$3.49"), ("BREAD", "$2.19"), ("COFFEE", "$8.99"), ("SALE", "20% OFF")]
for index, (name, price) in enumerate(labels):
    y = 70 + index * 105
    draw.rounded_rectangle(
        (80, y, 688, y + 82), radius=10, fill="white", outline="#cc2222", width=4
    )
    draw.text((110, y + 21), name, fill="black", font=title_font)
    draw.text((470, y + 21), price, fill="#cc2222", font=title_font)
image.save(OUT / "03_ocr.jpg", quality=95)

image, draw = canvas("SCENE 4 - STORE ENVIRONMENT")
draw.rectangle((0, 70, 768, 512), fill="#d8d8d8")
draw.rectangle((25, 90, 185, 450), fill="#8b6f47")
draw.rectangle((583, 90, 743, 450), fill="#8b6f47")
draw.ellipse((270, 330, 475, 435), fill="#55aee6", outline="#176ca3", width=4)
draw.text((320, 365), "WATER", fill="white", font=text_font)
draw.rectangle((455, 250, 570, 365), fill="#a66a3f", outline="black", width=3)
draw.text((468, 290), "BOX", fill="white", font=text_font)
draw.text(
    (205, 465),
    "Wet floor and carton obstructing the aisle",
    fill="#b00020",
    font=text_font,
)
image.save(OUT / "04_environment.jpg", quality=95)

image, draw = canvas("SCENE 5 - OUT OF STOCK DETECTION")
shelves(draw)
filled = {(0, 0), (0, 5), (1, 2), (2, 6)}
for row in range(3):
    for col in range(7):
        x, y = 45 + col * 99, 65 + row * 150
        draw.rectangle((x, y, x + 70, y + 82), outline="#999999", width=2)
        if (row, col) in filled:
            draw.rectangle((x + 3, y + 3, x + 67, y + 79), fill="#3977c3")
            draw.text((x + 15, y + 27), "SKU", fill="white", font=text_font)
        else:
            draw.text((x + 8, y + 30), "EMPTY", fill="#cc2222")
image.save(OUT / "05_out_of_stock.jpg", quality=95)

for path in sorted(OUT.glob("*.jpg")):
    print(path)

from PIL import Image, ImageOps


def get_image_matrix(image_path: str, size: tuple[int, int]) -> list[float]:
    return parse_image(Image.open(image_path), size)


def parse_image(image: Image, size: tuple[int, int]):
    threshold: int = 128

    image = image.convert('L')
    image = ImageOps.invert(image)
    image = image.crop(image.getbbox())
    image = ImageOps.invert(image)

    max_dim: int = max(image.size)
    delta_w: int = max_dim - image.size[0]
    delta_h: int = max_dim - image.size[1]
    image = ImageOps.expand(
        image, (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2), fill=255)

    image = image.resize(size, Image.LANCZOS)
    return list(image.point(lambda p: 1.0 if p < threshold else 0.0).getdata())
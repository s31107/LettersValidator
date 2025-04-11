from PIL import Image


def get_image_matrix(path: str, matrix_width: int, matrix_height: int) -> list[int]:
    image: Image = Image.open(path).convert("L").resize((matrix_width, matrix_height))
    return [int(item == 0) for item in image.getdata()]

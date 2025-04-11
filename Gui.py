import tkinter
from PIL import Image, ImageDraw


def get_picture(matrix_width: int, matrix_height: int) -> list[int]:
    root = tkinter.Tk()
    app = LetterDrawer(root)
    root.mainloop()
    return app.get_image(matrix_width, matrix_height)


class LetterDrawer:
    _last_x: float = None
    _last_y: float = None
    _canvas_size: tuple[int, int] = (500, 500, )

    def __init__(self, root: tkinter.Tk):
        self._canvas: tkinter.Canvas = tkinter.Canvas(
            root, width=self._canvas_size[0], height=self._canvas_size[1], bg="white")
        self._canvas.pack()

        clear_button: tkinter.Button = tkinter.Button(root, text="Wyczyść", command=self.clear_canvas)
        accept_button: tkinter.Button = tkinter.Button(root, text="Potwierdź", command=root.destroy)
        accept_button.pack(side=tkinter.RIGHT, padx=5, pady=5)
        clear_button.pack(side=tkinter.RIGHT, padx=5, pady=5)

        self._canvas.bind("<B1-Motion>", self.draw)
        self._canvas.bind("<ButtonRelease-1>", lambda _: self.set_default_x_y())

        self._image = Image.new("L", self._canvas_size, color="white")
        self._image_draw = ImageDraw.Draw(self._image)

    def set_default_x_y(self) -> None:
        self._last_x: None = None
        self._last_y: None = None

    def draw(self, event) -> None:
        if self._last_x is not None and self._last_y is not None:
            self._canvas.create_line(self._last_x, self._last_y, event.x, event.y,
                                     width=6, fill="black", capstyle=tkinter.ROUND, smooth=True)
            self._image_draw.line([self._last_x, self._last_y, event.x, event.y])
        self._last_x: float = event.x
        self._last_y: float = event.y

    def clear_canvas(self) -> None:
        self._canvas.delete("all")
        self.set_default_x_y()

    def get_image(self, width: int, height: int) -> list[int]:
        return [int(item == 0) for item in self._image.resize((width, height, )).getdata()]

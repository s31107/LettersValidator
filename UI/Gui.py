import io
import tkinter as tk

from PIL import Image, ImageOps


def get_binary_pixels(canvas: tk.Canvas):
    image = Image.open(io.BytesIO(canvas.postscript(colormode="gray")))
    resized_image = image.convert("L").resize((100, 100))
    return [int(item == 0) for item in resized_image.getdata()]

class LetterDrawer:
    last_x, last_y = None, None
    def __init__(self, root):

        self.canvas = tk.Canvas(root, width=500, height=500, bg="white")
        self.canvas.pack()

        self.clear_button = tk.Button(root, text="Wyczyść", command=self.clear_canvas)
        self.accept_button = tk.Button(root, text="Potwierdź")
        self.accept_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.clear_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", lambda _: self.set_default_x_y())

    def set_default_x_y(self):
        self.last_x, self.last_y = None, None

    def draw(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=6, fill='black', capstyle=tk.ROUND, smooth=True)
        self.last_x = event.x
        self.last_y = event.y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.set_default_x_y()


root = tk.Tk()
app = LetterDrawer(root)
root.mainloop()
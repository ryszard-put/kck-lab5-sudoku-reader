from dataclasses import dataclass


@dataclass
class Transformation:
    image: any
    title: str
    grayscale: bool

    def plot(self, ax):
        ax.imshow(self.image, cmap="gray") if self.grayscale else ax.imshow(
            self.image)
        ax.set_title(self.title)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])

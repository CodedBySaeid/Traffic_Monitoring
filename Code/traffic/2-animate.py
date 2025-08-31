from pickletools import optimize
from PIL import Image
import os
import cv2

image_folder = 'Code/traffic/Data/detrac2/images/train'
image_files = sorted(
    [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))]
)

def makegif():
    # This function creates a gif out of the images. BUT the file size would be huge.
    output_gif = 'detrac.gif'

    images = [Image.open(os.path.join(image_folder, f)) for f in image_files]
    if images:
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=50,  # milliseconds between frames
            loop=1,         # 0 means loop forever
            optimize=True
        )
        print(f"GIF saved to: {output_gif}")
    else:
        print("No images found in the folder.")


def makevid():

    frame = cv2.imread(os.path.join(image_folder, image_files[0]))
    h, w, _ = frame.shape
    out = cv2.VideoWriter("detrac.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))

    for f in image_files:
        frame = cv2.imread(os.path.join(image_folder, f))
        out.write(frame)

    out.release()


makevid()

from PIL import Image
from glob import glob
import tqdm

paths = glob("*.jpg")
for path in tqdm(paths):
	img = Image.open(path)
	img_resize = img.resize((224, 224))
	img_resize.save("../image/" + path)


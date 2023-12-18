import os
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from src.models.modnet import MODNet
from IPython.display import display

# define input and output paths
input_folder = 'input'
output_folder = 'output'
ckpt_path = 'pretrained/modnet_photographic_portrait_matting.ckpt'

# create MODNet and load pre-trained checkpoint
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)
if torch.cuda.is_available():
    modnet = modnet.cuda()
    weights = torch.load(ckpt_path)
else:
    weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
modnet.load_state_dict(weights)
modnet.eval()

# define image to tensor transform
im_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
# clean and rebuild the output folder
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# iterate through input images and perform inference
image_names = os.listdir(input_folder)
for image_name in image_names:
    print('Process image: {0}'.format(image_name))

    # read image
    input_image = Image.open(os.path.join(input_folder, image_name))

    # convert image to PyTorch tensor
    im = im_transform(input_image)

    # add mini-batch dim
    im = im[None, :, :, :]

    # inference
    _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

    # save matte
    matte = F.interpolate(matte, size=(im.shape[2], im.shape[3]), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    matte_name = image_name.split('.')[0] + '.png'
    Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(output_folder, matte_name))


# define combined_display function
def combined_display(image, matte, image_name):
    # calculate display resolution
    w, h = image.width, image.height
    rw, rh = 800, int(h * 800 / (3 * w))

    # resize matte to the same size as image
    matte = matte.resize((image.width, image.height))

    # obtain predicted foreground
    image = np.asarray(image)
    if len(image.shape) == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, 0:3]
    matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
    foreground = image * matte + np.full(image.shape, 255) * (1 - matte)

    # combine image, foreground, and alpha into one line
    matte_name = image_name.split('.')[0] + '.png'
    combined = Image.fromarray(np.uint8(foreground)).save((os.path.join(output_folder, matte_name)))
    return combined

# visualize all images
image_names = os.listdir(input_folder)
for image_name in image_names:
    matte_name = image_name.split('.')[0] + '.png'
    image = Image.open(os.path.join(input_folder, image_name))
    matte = Image.open(os.path.join(output_folder, matte_name))
    display(combined_display(image, matte, image_name))
    print(image_name, '\n')



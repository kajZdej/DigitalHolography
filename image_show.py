# to show images saved in tif format

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import PIL

def show_image(image_path):
    image = tiff.imread(image_path)
    print('Image shape:', image.shape)
    print('Image type:', type(image))
    #if there is any metadata, print it
    if 'ImageDescription' in tiff.TiffFile(image_path).pages[0].tags:
        print('Image metadata:', tiff.TiffFile(image_path).pages[0].tags['ImageDescription'].value)
        
    name = image_path.split('/')[-1]
    plt.imshow(image, cmap='gray')
    plt.savefig('images/'+name+'.png')
    return image
    
# teke midle 1000 * 1000 pixels
def take_midle(image_path, size=1000):
    image = tiff.imread(image_path)
    if len(image.shape) == 3:
        image = image[:, :, 0]
        # flop image
        image = np.flip(image, 1)
    x, y = image.shape
    x1 = x//2 - size//2
    x2 = x//2 + size//2
    y1 = y//2 - size//2
    y2 = y//2 + size//2
    image = image[x1:x2, y1:y2]
    image_max = np.max(image)
    image_min = np.min(image)
    print('Image max:', image_max)
    print('Image min:', image_min)
    image = ((image - image_min) / (image_max - image_min)) * 240
    image = image//1
    return image

def png_image(image_path):
    image = plt.imread(image_path)
    print('Image shape:', image.shape)
    print('Image type:', type(image))
    return image

def pari(path):
    image = png_image(path)
    # take half of image
    x, y = image.shape
    x1 = x//2
    y1 = y//2
    imageh = image[:, :x]
    imageb = image[:, -x:]
    # save as png
    plt.imsave('images/imageh.png', imageh, cmap='gray')
    plt.imsave('images/imageb.png', imageb, cmap='gray')

def posamezna(path):
    vzorec = take_midle(path, 500)
    #print(vzorec)
    vzorec = vzorec.astype(np.uint16)
    #save as tif
    tiff.imwrite('images/vzorec.tif', vzorec)
    # save as png
    plt.imsave('images/vzorec.png', vzorec, cmap='gray')
    print('Image saved as png')
    

if __name__ == '__main__':
    path = '/mnt/data/datasets/polen/raw/holograms/training/holo_pelod_training_vzorci_T1_2.tiff'
    posamezna(path)
    
    
    
    
    

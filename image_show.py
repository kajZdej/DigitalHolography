# to show images saved in tif format

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

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
def take_midle(image_path):
    image = tiff.imread(image_path)
    image_max = np.max(image)
    image_min = np.min(image)
    print('Image max:', image_max)
    print('Image min:', image_min)
    image = ((image - image_min) / (image_max - image_min)) * 240
    # to int
    image = image.astype(np.uint8)
    x, y = image.shape
    x1 = x//2 - 500
    x2 = x//2 + 500
    y1 = y//2 - 500
    y2 = y//2 + 500
    return image[x1:x2, y1:y2]

def png_image(image_path):
    image = plt.imread(image_path)
    print('Image shape:', image.shape)
    print('Image type:', type(image))
    return image


if __name__ == '__main__':
    path = '/mnt/data/datasets/polen/holografus/Meritve_HOLO_2/patches_test/patch_pair_38.png'
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
    
    
    # vzorec = take_midle(path)
    # print(vzorec)
    # save as tif
    #tiff.imwrite('images/vzorec.tif', vzorec)
    #plt.imshow(vzorec, cmap='gray')
    #plt.savefig('images/vzorec.png')
    #print('Image saved as png')
    

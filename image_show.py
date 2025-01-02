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
    

if __name__ == '__main__':
    path = '/home/antonb/DigitalHolography/samples/TS-20220402201632996.tif'
    show_image(path)
    print('Image saved as png')
    

# to show images saved in tif format

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

def show_image(image_path):
    image = tiff.imread(image_path)
    name = image_path.split('/')[-1]
    plt.imshow(image, cmap='gray')
    plt.savefig('images/'+name+'.png')
    

if __name__ == '__main__':
    path = '/home/antonb/DigitalHolography/samples/TS-20220310163413358.tif'
    show_image(path)
    print('Image saved as png')
    

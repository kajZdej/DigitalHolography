# -*- coding: utf-8 -*-


import torch  
import torch.nn as nn
import numpy as np
import PIL
from PIL import Image, ImageOps

import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"



print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
device_default = torch.cuda.current_device()
torch.cuda.device(device_default)
print(torch.cuda.get_device_name(device_default))
device = torch.device("cuda")
print(torch.version.cuda)
print(torch.__version__)
print(torch.cuda.get_arch_list())



img_size = 1000

'''
Spherical light function
Nx, Ny : hologram size
z : object-sensor distance 
wavelength: wavelength of light
deltaX, deltaY : sensor size
'''
# um
Nx = 1000 # pixels
Ny = 1000 # pixels
z = 1650 # mm
wavelength = 0.52 # um
deltaX = 2 # um
deltaY = 2 # um

'''

'''
#load image
img = Image.open('/mnt/data/home/antonn/SEMINAR/DigitalHolography/results_old/holo_1650_0.52_2022-03-15 00_36_24.bmp') # load image
#take just one chanel of tiff
#print(img.size) 
#print(type(img))
img = ImageOps.grayscale(img)
#print(img.size)
#print(type(img))
#save input image as png
img.save('./input/input.png') # save input image as png

h,w = img.size # get image size

# resize image to 1000x1000
if h != 1000 or w != 1000: 
    img = img.resize((1000,1000), PIL.Image.LANCZOS)

#img.show()
# img.save('./input/input.png') # save input image as png
# pytorch provides a function to convert PIL images to tensors.
pil2tensor = transforms.ToTensor() # save function to convert PIL images to tensors
tensor2pil = transforms.ToPILImage() # save function to convert tensors to PIL images

tensor_img = pil2tensor(img) # convert image to tensor

g = tensor_img.numpy() # convert tensor to numpy array
g = np.sqrt(g) # take square root of the image
g = (g-np.min(g))/(np.max(g)-np.min(g)) # normalize the image

plt.figure(figsize=(20,15)) # set figure size
print(np.squeeze(g).shape) # print shape of the image
# plt.imshow(np.squeeze(g), cmap='gray') # plot the image





'''
Phase Unwap and fft
'''
def unwrap(x):
    # calculate the phase from the complex number
    y = x % (2 * np.pi)
    return torch.where(y > np.pi, 2*np.pi - y, y)

def fft2dc(x):
    # return the 2D fourier transform of the input
    return np.fft.fftshift(np.fft.fft2(x))
  
def ifft2dc(x):
    # return the 2D inverse fourier transform of the input
    return np.fft.ifft2(np.fft.fftshift(x))

def Phase_unwrapping(in_):
    # in_ is the input image
    f = np.zeros((1000,1000)) # create an array of zeros
    for ii in range(1000): 
        for jj in range(1000): 
            x = ii - 1000/2 
            y = jj - 1000/2
            f[ii,jj] = x**2 + y**2 # make cilindrical mask (center of the image is 0 and the values increase as we move away from the center)
    # make the output image real
    a = ifft2dc(fft2dc(np.cos(in_)*ifft2dc(fft2dc(np.sin(in_))*f))/(f+0.000001)) 
    # make the output image real
    b = ifft2dc(fft2dc(np.sin(in_)*ifft2dc(fft2dc(np.cos(in_))*f))/(f+0.000001))
    out = np.real(a - b)
    return out




def propagator(Nx,Ny,z,wavelength,deltaX,deltaY):
    # Nx, Ny : hologram size
    # z : object-sensor distance
    # wavelength: wavelength of light
    # deltaX, deltaY : sensor size
    
    k = 1/wavelength # wave number
    x = np.expand_dims(np.arange(np.ceil(-Nx/2),np.ceil(Nx/2),1)*(1/(Nx*deltaX)),axis=0) # create an array of x values
    y = np.expand_dims(np.arange(np.ceil(-Ny/2),np.ceil(Ny/2),1)*(1/(Ny*deltaY)),axis=1) # create an array of y values
    y_new = np.repeat(y,Nx,axis=1) # repeat the y values
    x_new = np.repeat(x,Ny,axis=0) # repeat the x values
    kp = np.sqrt(y_new**2+x_new**2) # calculate the spatial frequency
    term=k**2-kp**2 # calculate the term
    term=np.maximum(term,0) # calculate the maximum term
    phase = np.exp(1j*2*np.pi*z*np.sqrt(term)) # calculate the phase
    return phase


'''
Back-propogation
'''
phase = propagator(Nx,Ny,z,wavelength,deltaX,deltaY) # calculate the phase
eta = np.fft.ifft2(np.fft.fft2(g)*np.fft.fftshift(np.conj(phase))) # calculate the back-propogation
plt.figure(figsize=(20,15)) # set figure size
plt.imshow(np.squeeze(np.abs(eta)), cmap='gray')  # plot the image
plt.close() # close the plot



new_holo = ifft2dc(np.fft.fft2(eta)*np.fft.fftshift(phase)) # calculate the new hologram
plt.figure(figsize=(20,15))
plt.imshow(np.squeeze(np.abs(new_holo)), cmap='gray')
plt.close()

bp = np.squeeze(np.abs(eta)) # calculate the back-propogation
bp = bp/(np.max(bp)-np.min(bp)) *255 # normalize the image


cv2.imwrite('./bp.png',bp) # save the back-propogation as a png file

# a = np.fft.fftshift(np.conj(phase))
# b = np.conj(phase)
# c = np.squeeze(np.abs(np.fft.fft2(g)))
# c = c/(np.max(c)-np.min(c)) * 255

# plt.imshow(np.abs(a))
# plt.imshow(np.abs(b))
# plt.imshow(np.squeeze(g))

# plt.imshow(kp)
# plt.imshow(np.squeeze(np.abs(phase)), cmap='gray')




'''
Define loss function


'''
class RECLoss(nn.Module):

    def __init__(self):
        super(RECLoss,self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.z = z
        self.wavelength = wavelength
        self.deltaX = deltaX
        self.deltaY = deltaY
        self.prop = self.propagator(self.Nx,self.Ny,self.z,self.wavelength,self.deltaX,self.deltaY)
        self.prop = self.prop.cuda()

    def propagator(self,Nx,Ny,z,wavelength,deltaX,deltaY):
        k = 1/wavelength
        x = np.expand_dims(np.arange(np.ceil(-Nx/2),np.ceil(Nx/2),1)*(1/(Nx*deltaX)),axis=0)
        y = np.expand_dims(np.arange(np.ceil(-Ny/2),np.ceil(Ny/2),1)*(1/(Ny*deltaY)),axis=1)
        y_new = np.repeat(y,Nx,axis=1)
        x_new = np.repeat(x,Ny,axis=0)
        kp = np.sqrt(y_new**2+x_new**2)
        term=k**2-kp**2
        term=np.maximum(term,0) 
        phase = np.exp(1j*2*np.pi*z*np.sqrt(term))
        return torch.from_numpy(np.concatenate([np.real(phase)[np.newaxis,:,:,np.newaxis], np.imag(phase)[np.newaxis,:,:,np.newaxis]], axis = 3))
   

    def roll_n(self, X, axis, n):
        f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    def batch_fftshift2d(self, x):
        real, imag = torch.unbind(x, -1)
        for dim in range(1, len(real.size())):
            n_shift = real.size(dim)//2
            if real.size(dim) % 2 != 0:
                n_shift += 1  # for odd-sized images
            real = self.roll_n(real, axis=dim, n=n_shift)
            imag = self.roll_n(imag, axis=dim, n=n_shift)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

    def batch_ifftshift2d(self,x):
        real, imag = torch.unbind(x, -1)
        for dim in range(len(real.size()) - 1, 0, -1):
            real = self.roll_n(real, axis=dim, n=real.size(dim)//2)
            imag = self.roll_n(imag, axis=dim, n=imag.size(dim)//2)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)
    
    def complex_mult(self, x, y):
        real_part = x[:,:,:,0]*y[:,:,:,0]-x[:,:,:,1]*y[:,:,:,1]
        real_part = real_part.unsqueeze(3)
        imag_part = x[:,:,:,0]*y[:,:,:,1]+x[:,:,:,1]*y[:,:,:,0]
        imag_part = imag_part.unsqueeze(3)
        return torch.cat((real_part, imag_part), 3)

    def forward(self, x, y):
        batch_size = x.size()[0]
        
        x = x.squeeze(2)
        y = y.squeeze(2)
        x = x.permute([0, 2, 3, 1])
        y = y.permute([0, 2, 3, 1])
        
        cEs = self.batch_fftshift2d(torch.fft.fftn(x, dim=(1, 2, 3), norm='ortho'))
        cEsp = self.complex_mult(cEs, self.prop)
        
        # forward propagate
        reconstrut_freq = cEsp
        reconstrut_freq_abs = torch.abs(reconstrut_freq)
        reconstrut_freq_abs = (reconstrut_freq_abs - torch.min(reconstrut_freq_abs)) / (torch.max(reconstrut_freq_abs) - torch.min(reconstrut_freq_abs))
        
        capture_freq = torch.log(torch.abs(self.batch_fftshift2d(torch.fft.fftn(y, dim=(1, 2, 3), norm='ortho'))) + 1e-5)
        capture_freq = (capture_freq - torch.min(capture_freq)) / (torch.max(capture_freq) - torch.min(capture_freq))
        
        h_x = x.size()[1]
        w_x = x.size()[2]
        
        h_tv_x = torch.pow((reconstrut_freq_abs[:, 1:, :, :] - reconstrut_freq_abs[:, :h_x-1, :, :]), 2).sum()
        w_tv_x = torch.pow((reconstrut_freq_abs[:, :, 1:, :] - reconstrut_freq_abs[:, :, :w_x-1, :]), 2).sum()
        
        h_tv_y = torch.pow((capture_freq[:, 1:, :, :] - capture_freq[:, :h_x-1, :, :]), 2).sum()
        w_tv_y = torch.pow((capture_freq[:, :, 1:, :] - capture_freq[:, :, :w_x-1, :]), 2).sum()
        
        count_h = self._tensor_size(x[:, 1:, :, :])
        count_w = self._tensor_size(x[:, :, 1:, :])
        
        tv_diff = 2 * (h_tv_x / count_h + w_tv_x / count_w) / batch_size - 2 * (h_tv_y / count_h + w_tv_y / count_w) / batch_size
        print(0.01 * tv_diff)

        S = torch.fft.ifftn(self.batch_ifftshift2d(cEsp), dim=(1, 2, 3), norm='ortho')
        Se = S[:,:,:,0]
        
        mse = torch.mean(torch.abs(Se-y[:,:,:,0]))/2-0.01*tv_diff
        return mse

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


'''
discrete wavelet transform
'''
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


'''
Define Network
'''
# finish the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv_init = nn.Sequential( 
            nn.Conv2d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
        )
        
        self.conv_1 = nn.Sequential(   
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        
        self.conv_2 = nn.Sequential(   
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        
        self.conv_nonlinear = nn.Sequential(   
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 16, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        
        
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(16, 1024, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
        )
        
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            #nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            #nn.BatchNorm2d(16),
            nn.Conv2d(16, 2, 3, stride=1, padding=1),
        )
        
    
    def forward(self,x):
        x = x.float()
        x = self.conv_init(x)
        x = dwt_init(x)
        x = self.conv_1(x)
        x = dwt_init(x)
        x = self.conv_2(x)
        x = dwt_init(x)
        x = self.conv_nonlinear(x)
        
        x = self.deconv_1(x)
        x = iwt_init(x)
        x = self.deconv_2(x)
        x = iwt_init(x)
        x = self.deconv_3(x)
        x = iwt_init(x)
        x = self.deconv_4(x)
        return x


from torchsummary import summary
criterion_1 = RECLoss()
model = Net().cuda()
optimer_1 = optim.Adam(model.parameters(), lr=1e-3)


device = torch.device("cuda")
epoch_1 = 5000
epoch_2 = 2000
period = 100
eta = torch.from_numpy(np.concatenate([np.real(eta)[np.newaxis,:,:], np.imag(eta)[np.newaxis,:,:]], axis = 1))
holo = torch.from_numpy(np.concatenate([np.real(g)[np.newaxis,:,:], np.imag(g)[np.newaxis,:,:]], axis = 1))


for i in range(epoch_1):
    in_img = eta.to(device)
    target = holo.to(device)
    
    out = model(in_img) 
    l1_loss = criterion_1(out, target)
    loss = l1_loss
    
    optimer_1.zero_grad()
    loss.backward()
    optimer_1.step()
    
    print('epoch [{}/{}]     Loss: {}'.format(i+1, epoch_1, l1_loss.cpu().data.numpy()))
    if ((i+1) % period) == 0:
        outtemp = out.cpu().data.squeeze(0).squeeze(1)
        outtemp = outtemp
        plotout = torch.sqrt(outtemp[0,:,:]**2 + outtemp[1,:,:]**2)
        plotout = (plotout - torch.min(plotout))/(torch.max(plotout)-torch.min(plotout))
        plt.figure(figsize=(20,15))
        plt.imshow(tensor2pil(plotout), cmap='gray')
        plt.savefig('./output/{}_amp.png'.format(i+1))
        #plt.show()
        
        plotout_p = (torch.atan(outtemp[1,:,:]/outtemp[0,:,:])).numpy()
        plotout_p = Phase_unwrapping(plotout_p)
        plotout_p = (plotout_p - np.min(plotout_p))/(np.max(plotout_p)-np.min(plotout_p))
        plt.figure(figsize=(20,15))
        plt.imshow((plotout_p), cmap='gray')
        #plt.show()
        
        
outtemp = out.cpu().data.squeeze(0).squeeze(1)
outtemp = outtemp
plotout = torch.sqrt(outtemp[0,:,:]**2 + outtemp[1,:,:]**2)
plotout = (plotout - torch.min(plotout))/(torch.max(plotout)-torch.min(plotout))
plt.figure(figsize=(30,30))
plt.imshow(tensor2pil(plotout), cmap='gray')
#plt.savefig('./output/{}_amp.png'.format(i+1))
#plt.show()


plotout_p = (torch.atan(outtemp[1,:,:]/outtemp[0,:,:])).numpy()
plotout_p = Phase_unwrapping(plotout_p)
plotout_p = (plotout_p - np.min(plotout_p))/(np.max(plotout_p)-np.min(plotout_p))
plt.figure(figsize=(30,30))
plt.imshow((plotout_p), cmap='gray')
#plt.savefig('./output/{}_phase.png'.format(i+1))
#plt.show()        





#cv2.imwrite("./penalty_1/1_amp.png",tensor2pil(plotout))

torch.__version__


type(tensor2pil(plotout))

amp =tensor2pil(plotout)
amp.save("./output/1_amp.png")



import cv2
cv2.imwrite("./output/phase.png",plotout_p)

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import torch
import cv2

def take_middle(image, size=1000):
    x, y = image.shape
    x1 = max(0, x//2 - size//2)
    x2 = min(x, x//2 + size//2)
    y1 = max(0, y//2 - size//2)
    y2 = min(y, y//2 + size//2)
    return image[x1:x2, y1:y2]

def show_image(image_path):
    image = tiff.imread(image_path)
    print('Image shape:', image.shape)
    print('Image type:', type(image))
    if 'ImageDescription' in tiff.TiffFile(image_path).pages[0].tags:
        print('Image metadata:', tiff.TiffFile(image_path).pages[0].tags['ImageDescription'].value)
        
    name = image_path.split('/')[-1]
    plt.imshow(image, cmap='gray')
    plt.savefig('images/'+name+'.png')
    return image

# Example usage
image_path = 'path/to/your/image.tif'
image = show_image(image_path)
masked_image = take_middle(image, size=min(image.shape))
plt.imshow(masked_image, cmap='gray')
plt.savefig('images/masked_image.png')

# Save tensor as image
plotout = torch.tensor(masked_image)
plotout = (plotout - torch.min(plotout)) / (torch.max(plotout) - torch.min(plotout))
amp = tensor2pil(plotout)
amp.save("./output/1_amp.png")

plotout_p = (torch.atan(outtemp[1, :, :] / outtemp[0, :, :])).numpy()
plotout_p = Phase_unwrapping(plotout_p)
plotout_p = (plotout_p - np.min(plotout_p)) / (np.max(plotout_p) - np.min(plotout_p))
cv2.imwrite("./output/phase.png", plotout_p)










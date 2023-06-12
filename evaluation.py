import numpy as np
import lpips
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

import argparse
import os
import lpips

#Imports for SSIM/RMSE
from skimage import io


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('-d0','--dir0', type=str, default='./input')
#parser.add_argument('-d1','--dir1', type=str, default='./total3d')

parser.add_argument('-i', '--input', type=str, default='./input')
parser.add_argument('-t', '--total3d', type=str, default='./total3d')
parser.add_argument('-im', '--im3d', type=str, default='./im3d')
parser.add_argument('-ipf', '--instpifu', type=str, default='./instpifu')
parser.add_argument('-o','--out', type=str, default='./output/matrix.txt')

parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.input)

for file in files:
	if(os.path.exists(os.path.join(opt.total3d,file))) and (os.path.exists(os.path.join(opt.instpifu, file))):
		# Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.input,file))) # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.total3d,file)))
		print(os.path.join(opt.instpifu, file))
		img2 = lpips.im2tensor(lpips.load_image(os.path.join(opt.instpifu,file)))
		img3 = lpips.im2tensor(lpips.load_image(os.path.join(opt.im3d,file)))
		inputdir = os.path.join(opt.input, file)
		total3ddir = os.path.join(opt.total3d, file)
		ipfudir = os.path.join(opt.instpifu, file)
		im3dir = os.path.join(opt.im3d, file)
		skimg0 = (io.imread(inputdir))[..., 0:3]
		skimg1 = io.imread(total3ddir)
		skimg2 = io.imread(ipfudir)
		skimg3 = io.imread(im3dir)
		print(file)

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()
			img2 = img2.cuda()
			img3 = img3.cuda()

		# Compute distance
		lpipst3d = loss_fn.forward(img0,img1)
		lpipsipfu = loss_fn.forward(img0,img2)
		lpipsim3d = loss_fn.forward(img0, img3)
		print('LPIPS: %s: %.3f, %.3f, %.3f' %(file,lpipst3d,lpipsipfu,lpipsim3d))
		f.writelines('LPIPS: %s: %.6f, %.6f, %.6f\n'%(file,lpipst3d,lpipsipfu,lpipsim3d))
		ssimt3d = ssim(skimg0, skimg1, data_range=skimg1.max() - skimg1.min(), channel_axis=2)
		ssimipfu = ssim(skimg0, skimg2, data_range=skimg2.max() - skimg2.min(), channel_axis=2)
		ssim3d = ssim(skimg0, skimg3, data_range=skimg3.max() - skimg3.min(), channel_axis=2)
		print('SSIM: %s: %.3f, %.3f, %.3f' %(file,ssimt3d,ssimipfu,ssim3d))
		f.writelines('SSIM: %s: %.6f, %.6f, %.6f\n'%(file,ssimt3d,ssimipfu, ssim3d))
		rmset3d = mean_squared_error(skimg0, skimg1)**(0.5)
		rmseipfu = mean_squared_error(skimg0, skimg2)**(0.5)
		rmseim3d = mean_squared_error(skimg0, skimg3)**(0.5)
		print('RMSE: %s: %.3f, %.3f, %.3f' %(file,rmset3d,rmseipfu, rmseim3d))
		f.writelines('RMSE: %s: %.6f, %.6f, %.6f\n'%(file,rmset3d,rmseipfu, rmseim3d))
        
f.close()



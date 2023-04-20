import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import warnings
import torch
import os
import tqdm
from skimage.morphology import disk, dilation
import glob
import argparse
import time
import platform
from os.path import join, isfile
import os
import sys
warnings.filterwarnings("ignore")

nib.Nifti1Header.quaternion_threshold = -100


# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))

model_path = join(application_path, 'models')
# print(model_path)
os.makedirs(model_path, exist_ok=True)


def download(url, file_name):
    import urllib.request
    import certifi
    import shutil
    import ssl
    context = ssl.create_default_context(cafile=certifi.where())
    #urllib.request.urlopen(url, cafile=certifi.where())
    with urllib.request.urlopen(url,
                                context=context) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)


def get_model(f):

    if isfile(f):
        return f
    
    model_file = join(model_path, f)
    
    if not os.path.exists(model_file):
        

        try:
            print(f'Downloading model files....')
            model_url = 'https://github.com/htylab/COMB/releases/download/model/COMB.pt'
            print(model_url, model_file)
            download(model_url, model_file)
            download_ok = True
            print('Download finished...')
        except:
            download_ok = False

        if not download_ok:
            raise ValueError('Server error. Please check the model name or internet connection.')
                
    return model_file

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d((2, 2, 1)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=(2, 2, 1),
                                  mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = torch.nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        return self.conv(x)


class UNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask
    

def remove_common_substrings(strings):
    # Split each string by os.sep
    split_strings = [s.split(os.sep) for s in strings]

    # Find the shortest split string in the list
    shortest_split_string = min(split_strings, key=len)

    # Initialize common substrings list
    common_substrings = []

    # Iterate through the elements of the shortest split string
    for i in range(len(shortest_split_string)):
        # Check if the element is common in all split strings
        element = shortest_split_string[i]
        if all(element in string for string in split_strings):
            common_substrings.append(element)

    # Remove the common substrings from each split string
    result = []
    for string in split_strings:
        new_string = [part for part in string if part not in common_substrings]
        result.append(new_string)

    # Join the parts of the strings using os.sep
    return ["_".join(parts) for parts in result]





def normalize(data):
    data_min = np.min(data)
    return (data - data_min) / (np.max(data) - data_min)


def seperate(img):
    emp = img * 0

    emp[img == 1] = 1
    lv = emp

    emp = img * 0
    emp[img == 2] = 2
    myo = emp

    emp = img * 0
    emp[img == 3] = 3
    rv = emp

    emp = img * 0
    emp[img == 4] = 4
    c = emp

    return lv, myo, rv, c


def recover(img, lv, myo, rv, curve):

    emp = img.copy()
    lv_d = lv * 0
    rv_d = rv * 0

    for ii in range(img.shape[-1]):
        lv_d[..., ii] = dilation(lv[..., ii], disk(2))
        rv_d[..., ii] = dilation(rv[..., ii], disk(2))

    emp[np.logical_and(lv_d == 1, curve == 4)] = 1
    emp[np.logical_and(rv_d == 3, curve == 4)] = 3
#     emp[rv_d==3] = 3
    emp[emp == 4] = 0
    emp[myo == 2] = 2

    return emp




def run(f, savef):

    vol = nib.load(f).get_fdata()

    affine = nib.load(f).affine
    zoom = nib.load(f).header.get_zooms()

    emp = vol * 0

    for jj in tqdm.tqdm(range(vol.shape[-2])):

        vol_nm = normalize(vol)[:, :, jj]

        vol_stack = np.stack([vol_nm, vol_nm, vol_nm, vol_nm])

        vol_d = torch.from_numpy(vol_stack[None, ...]).to(device).float()
        logits = NET(vol_d)
        softmax = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
        mask_pred = np.argmax(softmax[0, ...], axis=0)

        a, b, c, d = seperate(mask_pred)
        pred_dil = recover(mask_pred, a, b, c, d)  # 4d

        pred = np.array(pred_dil, dtype=np.uint8)

        emp[:, :, jj] = pred

    result = nib.Nifti1Image(emp, affine)
    result.header.set_zooms(zoom)

    nib.save(result, savef)

    return 1

device = 'cpu'
model_file = get_model('COMB.pt')
NET = torch.load(model_file, map_location='cpu')
state_dict = NET.state_dict()
NET = UNet3d(in_channels=4, n_classes=5, n_channels=24).to(device)
r = NET.load_state_dict(state_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+", type=str, help="Input mat file")
    parser.add_argument("-o", "--output", default=None, help="Output Path")
    args = parser.parse_args()

    print("Starting CMR COMB model.....")

    torch.cuda.empty_cache()



    # if the input file is a folder or contains an asterisk
    # use the glob function to find all input files.

    ffs = args.input
    if os.path.isdir(args.input[0]):
        ffs = glob.glob(os.path.join(args.input[0], "*.nii.gz"))

    elif "*" in args.input[0]:
        ffs = glob.glob(args.input[0])

    short_ffs = remove_common_substrings(ffs)
    print(f"Total files: {len(ffs)}")
    for ii in range(len(ffs)):
        f = ffs[ii]
        short_f = short_ffs[ii]

        # The following is for preparing the output directory
        # If the user provides an output directory and it does not exist, create it for them
        # If not provided, the default output directory is the folder of the input file

        f_output_dir = args.output

        if f_output_dir is None:
            f_output_dir = os.path.dirname(os.path.abspath(f))
        else:
            os.makedirs(f_output_dir, exist_ok=True)

        if len(ffs) > 1:
            f_output = short_f.replace(".nii.gz", "_mask.nii.gz")
        else:
            f_output = os.path.basename(f).replace(".nii.gz", "_mask.nii.gz")
        ff_output = os.path.join(f_output_dir, f_output)

        print(f"{ii + 1}: Processing {f}.....")
        # to create shorter filename for multiple mat files

        t = time.time()
        
        run(f, ff_output)


        print("Writing output:", ff_output)
        print("Processing time: %d seconds" % (time.time() - t))



if __name__ == "__main__":
    main()
    if platform.system() == "Windows":
        os.system("pause")

import os
import json
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# We are going to implement a UNet model for lane segmentation
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_c, out_c, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_c, out_c)
        )
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size = 2, stride = 2)
        self.conv = DoubleConv(out_c + skip_c, out_c)
    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([skip, x], dim = 1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 1):
        super().__init__()
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        self.down4 = Down(512,512)
        self.up1   = Up(512, 512, 256)
        self.up2   = Up(256, 256, 128)
        self.up3   = Up(128, 128, 64)
        self.up4   = Up(64,   64,  64)
        self.outc  = nn.Conv2d(64, n_classes, kernel_size = 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)

# We ae going to convert LLAMAS JSON annotations to a binary mask
def llamas_json_to_mask(json_path, img_h = 720, img_w = 1280):
    data = json.load(open(json_path))
    lanes = {l['lane_id']: l['markers'] for l in data.get('lanes', [])}
    if 'l0' not in lanes or 'r0' not in lanes:
        return np.zeros((img_h, img_w), dtype = np.uint8)
    def collect_pts(markers):
        pts = [(m['pixel_start']['x'], m['pixel_start']['y']) for m in markers]
        end = markers[-1]['pixel_end']
        pts.append((end['x'], end['y']))
        return pts
    left_pts = collect_pts(lanes['l0'])
    right_pts = collect_pts(lanes['r0'])
    poly = np.array(left_pts + right_pts[::-1], dtype = np.int32)
    mask = np.zeros((img_h, img_w), dtype = np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask

# We are going to create a custom dataset for the LLAMAS dataset
class LLAMASDataset(Dataset):
    def __init__(self, root_dir, split='train', img_transform = None, mask_transform = None):
        self.img_dir = os.path.join(root_dir, 'color_images', split)
        self.json_dir = os.path.join(root_dir, 'labels', split)
        self.ids = [f[:-5] for f in os.listdir(self.json_dir) if f.endswith('.json')]
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        img_file = next((f for f in os.listdir(self.img_dir) if f.startswith(id)), None)
        if not img_file:
            raise FileNotFoundError(f"No image for ID {id}")
        img = Image.open(os.path.join(self.img_dir, img_file)).convert('RGB')
        mask = Image.fromarray(llamas_json_to_mask(os.path.join(self.json_dir, id + '.json')))
        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, mask

# We will compute IoU and accuracy for validation
def compute_iou(preds, masks):
    preds = (torch.sigmoid(preds) > 0.5).float()
    intersection = (preds * masks).sum((1, 2, 3))
    union = preds.sum((1, 2, 3)) + masks.sum((1, 2, 3)) - intersection
    return (intersection / union.clamp(min = 1e-6)).mean().item()

def compute_accuracy(preds, masks):
    preds = (torch.sigmoid(preds) > 0.5).float()
    return (preds == masks).float().mean().item()

def train_with_validation(root_dir, weights_out, device):
    tf_img = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tf_mask = transforms.Compose([
        transforms.Resize((256,256), interpolation = Image.NEAREST),
        transforms.ToTensor()
    ])

    train_set = LLAMASDataset(root_dir, 'train', tf_img, tf_mask)
    valid_set = LLAMASDataset(root_dir, 'valid', tf_img, tf_mask)
    train_loader = DataLoader(train_set, batch_size = 16, shuffle = True)
    valid_loader = DataLoader(valid_set, batch_size = 16)

    model = UNet().to(device)
    if os.path.exists(weights_out):
        model.load_state_dict(torch.load(weights_out, map_location=device))

    optim = torch.optim.Adam(model.parameters(), lr = 1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')

    for epoch in range(1, 51):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss/(loop.n+1))

        # We will validate the model after each epoch
        model.eval()
        val_loss = 0
        iou_score = 0
        acc_score = 0
        with torch.no_grad():
            for imgs, masks in valid_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_loss += loss_fn(preds, masks).item()
                iou_score += compute_iou(preds, masks)
                acc_score += compute_accuracy(preds, masks)
        val_loss /= len(valid_loader)
        iou_score /= len(valid_loader)
        acc_score /= len(valid_loader)

        print(f"\n Epoch {epoch} - Val Loss: {val_loss:.4f}, IoU: {iou_score:.4f}, Acc: {acc_score:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_out)
            print(f" Best model saved with val loss {val_loss:.4f}\n")

# We are going to test running the inference on a input video
def segment_video(video_path, weights_path, device, input_size = (256,256), thresh = 0.5):
    model = UNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location = device))
    model.eval()
    tf_img = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret: break
        orig = frame.copy()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inp = tf_img(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            mask = torch.sigmoid(out)[0,0].cpu().numpy()
        mask = (mask > thresh).astype(np.uint8)*255
        mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))
        overlay = cv2.addWeighted(orig, 1.0, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        cv2.imshow('Segmentation', overlay)
        if cv2.waitKey(1)==ord('q'): break
    cap.release(); cv2.destroyAllWindows()

# This script is for training or running inference a UNet model on the LLAMAS dataset for lane segmentation
if __name__ == '__main__':
    MODE = 'train' 
    DATASET_PATH = r"C:/Users/minh tran/lane/llamas"
    VIDEO_PATH   = r"C:/Users/minh tran/dataassesst3/NO20250328-161610-000037.MP4"
    WEIGHTS_PATH = r"C:/Users/minh tran/lane/unet.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Running on: {device}")

    if MODE == 'train':
        train_with_validation(DATASET_PATH, WEIGHTS_PATH, device)
    else:
        segment_video(VIDEO_PATH, WEIGHTS_PATH, device)
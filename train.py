"""VOC binary segmentation training and evaluation."""
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from metrics import BinarySegmentationMetrics
from benchmark import benchmark_model

class VOC(Dataset):
    def __init__(self, root, split, size, encoding, augment=False, foreground_class=0):
        self.root, self.size, self.encoding, self.augment = Path(root), size, encoding, augment
        self.foreground_class = foreground_class
        paths = [self.root/'ImageSets'/'Segmentation'/f'{split}.txt', self.root/'ImageSets'/f'{split}.txt']
        manifest = next((p for p in paths if p.exists()), None)
        if manifest is None: raise FileNotFoundError(f'Missing split: {paths}')
        self.ids = [x.strip() for x in manifest.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
        if not self.ids: raise ValueError(f'Empty split: {manifest}')
        if len(set(self.ids)) != len(self.ids): raise ValueError('Duplicate image IDs')
        self.images = {}
        for p in (self.root/'JPEGImages').iterdir():
            if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}:
                if p.stem in self.images: raise ValueError(f'Ambiguous image stem: {p.stem}')
                self.images[p.stem] = p
        for name in self.ids:
            if name not in self.images or not (self.root/'SegmentationClass'/f'{name}.png').exists():
                raise FileNotFoundError(f'Missing image/mask: {name}')
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        name = self.ids[i]
        image = Image.open(self.images[name]).convert('RGB')
        mask = Image.open(self.root/'SegmentationClass'/f'{name}.png')
        if image.size != mask.size: raise ValueError(f'Image/mask size mismatch: {name}')
        raw = np.asarray(mask)
        if raw.ndim != 2: raise ValueError(f'Mask must be single-channel or palette: {name}')
        allowed = {0,1,255} if self.encoding == 'voc' else {0,255}
        if not set(np.unique(raw)).issubset(allowed): raise ValueError(f'Unexpected mask labels: {name}')
        image = image.resize((self.size,self.size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.size,self.size), Image.Resampling.NEAREST)
        if self.augment:
            if random.random()<.5: image,mask = ImageOps.mirror(image),ImageOps.mirror(mask)
            if random.random()<.5: image,mask = ImageOps.flip(image),ImageOps.flip(mask)
        raw = np.asarray(mask).copy()
        valid = raw != 255 if self.encoding=='voc' else np.ones_like(raw,dtype=bool)
        target = raw == (self.foreground_class if self.encoding=='voc' else 255)
        x = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
        x = (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        return x,torch.from_numpy(target).float()[None],torch.from_numpy(valid)[None]

def loss_fn(logits, target, valid):
    v=valid.float()
    bce=(torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*v).sum()/v.sum().clamp_min(1)
    p=logits.sigmoid()*v; y=target*v
    dice=1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)
    return bce+dice

@torch.no_grad()
def evaluate(model,loader,device,boundary_tolerance=2.0):
    model.eval()
    meter=BinarySegmentationMetrics(foreground_class=0,background_class=1,ignore_index=255,boundary_tolerance=boundary_tolerance)
    for x,y,v in loader:
        foreground=(model(x.to(device)).sigmoid()>=.5).cpu().squeeze(1)
        # Training uses foreground probability; uploaded metrics uses class 0 foreground.
        prediction=(~foreground).long()
        target=(~y.squeeze(1).bool()).long()
        target[~v.squeeze(1)]=255
        meter.update(prediction,target)
    return meter.compute()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True); p.add_argument('--output',default='outputs/unet')
    p.add_argument('--arch',default='Unet'); p.add_argument('--encoder',default='resnet34')
    p.add_argument('--weights',default='none',help='none (default, no downloads) or imagenet')
    p.add_argument('--mask-encoding',choices=['voc','binary255'],required=True,help='voc: 0/1 classes, 255 ignore; binary255: 0 background, 255 plant')
    p.add_argument('--foreground-class',type=int,choices=[0,1],default=0,help='VOC plant class; default 0 matches supplied metrics.py')
    p.add_argument('--boundary-tolerance',type=float,default=2.0)
    p.add_argument('--train-split',default='train'); p.add_argument('--val-split',default='val'); p.add_argument('--test-split',default='test')
    p.add_argument('--size',type=int,default=512); p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=8); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--benchmark-warmup',type=int,default=50)
    p.add_argument('--benchmark-iterations',type=int,default=200)
    p.add_argument('--resume',action='store_true'); p.add_argument('--eval-only',action='store_true')
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def dataset(split,aug=False): return VOC(a.data_root,split,a.size,a.mask_encoding,aug,a.foreground_class)
    train,val=dataset(a.train_split,True),dataset(a.val_split)
    if set(train.ids)&set(val.ids): raise ValueError('Train/val overlap')
    test=None
    if a.test_split.lower()!='none':
        test=dataset(a.test_split)
        if set(test.ids)&(set(train.ids)|set(val.ids)): raise ValueError('Test overlaps train/val')
    def loader(ds,shuffle=False): return DataLoader(ds,batch_size=a.batch_size,shuffle=shuffle,num_workers=a.workers,pin_memory=device.type=='cuda')
    restored=a.resume or a.eval_only
    model=smp.create_model(a.arch,encoder_name=a.encoder,encoder_weights=None if restored or a.weights.lower()=='none' else a.weights,in_channels=3,classes=1).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start,best=0,-1
    if restored:
        ck=torch.load(out/('best.pth' if a.eval_only else 'last.pth'),map_location=device,weights_only=False)
        for key in ['arch','encoder','size','mask_encoding','seed','epochs','lr','batch_size','data_root','train_split','val_split','test_split','weights','foreground_class','boundary_tolerance']:
            if ck['config'][key]!=vars(a)[key]: raise ValueError(f'Checkpoint config mismatch: {key}')
        model.load_state_dict(ck['model'])
        if a.resume and not a.eval_only:
            opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
            start,best=ck['epoch']+1,ck['best']
            random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng'].cpu())
            if device.type=='cuda' and ck['cuda_rng'] is not None: torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    elif (out/'last.pth').exists(): raise FileExistsError('Output exists: use --resume or a different --output')
    (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    trainloader,valloader=loader(train,True),loader(val)
    for epoch in range(start,a.epochs) if not a.eval_only else []:
        model.train(); total=0; begin=time.time()
        for x,y,v in trainloader:
            x,y,v=x.to(device),y.to(device),v.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                loss=loss_fn(model(x),y,v)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=loss.item()*len(x)
        result=evaluate(model,valloader,device,a.boundary_tolerance); scheduler.step()
        improved=result['miou']>best
        if improved: best=result['miou']
        ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch,best=best,config=vars(a),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        torch.save(ck,out/'last.pth')
        if improved: torch.save(ck,out/'best.pth')
        row=dict(epoch=epoch+1,loss=total/len(train),seconds=time.time()-begin,**result)
        with (out/'history.csv').open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=row.keys())
            if f.tell()==0:w.writeheader()
            w.writerow(row)
        print(json.dumps(row),flush=True)
    ck=torch.load(out/'best.pth',map_location=device,weights_only=False); model.load_state_dict(ck['model'])
    result=dict(arch=a.arch,encoder=a.encoder,seed=a.seed,best_epoch=ck['epoch']+1,parameters=sum(p.numel() for p in model.parameters()),val=evaluate(model,valloader,device,a.boundary_tolerance))
    if test is not None: result['test']=evaluate(model,loader(test),device,a.boundary_tolerance)
    result['benchmark']=benchmark_model(model,a.size,a.benchmark_warmup,a.benchmark_iterations)
    (out/'benchmark.json').write_text(json.dumps(result['benchmark'],indent=2))
    (out/'metrics.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()

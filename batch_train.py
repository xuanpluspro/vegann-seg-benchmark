"""Run experiments sequentially in isolated processes; collect metrics."""
import argparse,csv,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiments',default='experiments.json'); p.add_argument('--output-root',default='outputs'); p.add_argument('--seeds',type=int,nargs='+',default=[42]); p.add_argument('--resume',action='store_true')
    a,extra=p.parse_known_args(); root=Path(a.output_root); root.mkdir(parents=True,exist_ok=True)
    jobs=json.loads(Path(a.experiments).read_text()); rows=[]
    for job in jobs:
        for seed in a.seeds:
            out=root/f"{job['name']}_seed{seed}"; out.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,str(Path(__file__).with_name('train.py')),*extra,'--arch',job['arch'],'--encoder',job['encoder'],'--seed',str(seed),'--output',str(out)]
            if 'lr' in job: cmd+=['--lr',str(job['lr'])]
            if a.resume and (out/'last.pth').exists():cmd+=['--resume']
            print('Running:',cmd,flush=True)
            with (out/'console.log').open('a') as f: status=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
            row=dict(name=job['name'],seed=seed,status=status)
            if status==0:
                m=json.loads((out/'metrics.json').read_text()); row.update(parameters=m['parameters'],best_epoch=m['best_epoch'])
                for split in ['val','test']:
                    for k,v in m.get(split,{}).items(): row[f'{split}_{k}']=v
            rows.append(row)
            fields=list(dict.fromkeys(k for r in rows for k in r))
            with (root/'results.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
            print(f"Finished {out}: status={status}; see console.log",flush=True)
    if any(r['status'] for r in rows):sys.exit(1)
if __name__=='__main__': main()

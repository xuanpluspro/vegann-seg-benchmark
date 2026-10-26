"""Run experiments sequentially in isolated processes; collect metrics."""
import argparse,csv,json,subprocess,sys
from pathlib import Path
from datetime import datetime
from segmentation_metrics import result_rows, write_summary

def run_logged(cmd, log_path):
    # Binary forwarding preserves tqdm carriage returns in the terminal.
    with log_path.open('ab') as log:
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            while True:
                chunk=proc.stdout.read1(4096)
                if not chunk: break
                log.write(chunk); log.flush()
                sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            raise
        finally:
            proc.stdout.close()

def prepare_output(args, extra, parser):
    if args.legacy_output:
        root=Path(args.output_root)
    else:
        config=argparse.ArgumentParser(add_help=False)
        config.add_argument('--data-root',required=True)
        config.add_argument('--epochs',default='30')
        config.add_argument('--batch-size',default='32')
        config.add_argument('--weights',default='none')
        settings,_=config.parse_known_args(extra)
        dataset=args.dataset_name or Path(settings.data_root).name
        if args.resume and not args.run_name:
            parser.error('--resume requires --run-name to select the original experiment; old layouts use --legacy-output')
        run=args.run_name or f"{settings.weights}_ep{settings.epochs}_bs{settings.batch_size}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        for value in (dataset,run):
            if not value or value in {'.','..'} or '/' in value or '\\' in value:
                parser.error('dataset-name and run-name must be single folder names')
        root=Path(args.output_root)/dataset/run
        if args.resume and not root.is_dir():
            parser.error(f'Experiment does not exist: {root}')
        if not args.resume and root.exists() and any(root.iterdir()):
            parser.error(f'Experiment already exists: {root}; use --resume or a new --run-name')
    root.mkdir(parents=True,exist_ok=True)
    print(f'Experiment directory: {root.resolve()}',flush=True)
    return root

def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiments',default='experiments.json'); p.add_argument('--output-root',default='outputs'); p.add_argument('--seeds',type=int,nargs='+',default=[42]); p.add_argument('--resume',action='store_true')
    p.add_argument('--dataset-name',help='Dataset folder name; defaults to data-root basename')
    p.add_argument('--run-name',help='Experiment name; defaults to weights/epochs/batch/time')
    p.add_argument('--legacy-output',action='store_true',help='Use the old flat layout to resume existing experiments')
    a,extra=p.parse_known_args(); root=prepare_output(a,extra,p)
    jobs=json.loads(Path(a.experiments).read_text()); rows=[]
    total_jobs=len(jobs)*len(a.seeds); job_number=0
    for job in jobs:
        for seed in a.seeds:
            job_number+=1
            print(f'\nExperiment {job_number}/{total_jobs}: {job["name"]}, seed={seed}',flush=True)
            out=root/f"{job['name']}_seed{seed}"; out.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,'-u',str(Path(__file__).with_name('train.py')),*extra,'--arch',job['arch'],'--encoder',job['encoder'],'--seed',str(seed),'--output',str(out)]
            if 'lr' in job: cmd+=['--lr',str(job['lr'])]
            if a.resume and (out/'last.pth').exists():cmd+=['--resume']
            print('Running:',cmd,flush=True)
            status=run_logged(cmd,out/'console.log')
            if status==0:
                metrics=json.loads((out/'metrics.json').read_text())
                rows.extend(result_rows(metrics,name=job['name'],seed=seed,status=status))
            else:
                rows.append(dict(name=job['name'],seed=seed,status=status))
            write_summary(rows,root/'results.csv')
            print(f"Finished {out}: status={status}; see console.log",flush=True)
    if any(r['status'] for r in rows):sys.exit(1)
if __name__=='__main__': main()


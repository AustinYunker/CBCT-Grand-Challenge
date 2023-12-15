# torchrun --standalone --nnodes=1 --nproc_per_node=8  ./main.py

from model import unet
import torch, argparse, os, time, sys, shutil, logging, yaml
from data import CBCTDataset
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from matplotlib import pyplot as plt
import torchvision.transforms as T
from copy import deepcopy

def main(args):

    #Distributed training with Pytorch
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_nccl_available():
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    #output directory
    odir = args.expName + '-itrOut'
    if rank == 0:
        if os.path.isdir(odir):
            shutil.rmtree(odir)
        os.mkdir(odir)
        if os.path.isdir(f'{odir}/model_states') and os.path.isdir(f'{odir}/results'):
            shutil.rmtree(f'{odir}/model_states')
            shutil.rmtree(f'{odir}/results')
        os.mkdir(f'{odir}/model_states')
        os.mkdir(f'{odir}/results')

    torch.distributed.barrier()

    logging.basicConfig(filename=f'{args.expName}-itrOut/Noise2Inverse.log', level=logging.DEBUG)
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(f"local rank {local_rank} (global rank {rank}) of a world size {world_size} started")

    torch.cuda.set_device(local_rank)

    logging.info("\nLoading data into CPU memory, it will take a while ... ...")

    ds_train = CBCTDataset(ifn=args.th5, psz=args.psz)
    train_sampler = DistributedSampler(dataset=ds_train, shuffle=True, drop_last=True)
    dl_train = DataLoader(dataset=ds_train, batch_size=args.mbsz, sampler=train_sampler,\
                          num_workers=4, prefetch_factor=args.mbsz, drop_last=False, pin_memory=True)
    logging.info(f"\nLoaded %d samples, {ds_train.dim}, into CPU memory for training." % (len(ds_train), ))

    #load the validation data into the last GPU
    if rank == world_size-1:
        time.sleep(1)
        ds_valid = CBCTDataset(ifn=args.vh5, psz=args.psz)
        dl_valid = DataLoader(dataset=ds_valid, batch_size=args.mbsz, shuffle=False,\
                          num_workers=8, prefetch_factor=args.mbsz, drop_last=False, pin_memory=True)
        logging.info(f"Loaded %d samples, {ds_valid.dim}, into CPU memory for validation." % (len(ds_valid), ))

    #Option to train from scratch or pre-trained model
    if args.mdl is not None:
        checkpoint = torch.load(args.mdl, map_location=torch.device('cpu'))
        model = unet(start_filter_size=4)
        model.load_state_dict(checkpoint['model_state_dict']).cuda()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    else:
        model = unet(start_filter_size=4).cuda()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    criterion = torch.nn.MSELoss()
    #criterion = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    #data augmentation
    inp_transformer = T.RandomApply(transforms=[
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        ], p=.7)

    gt_transformer = T.RandomApply(transforms=[
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        ], p=.7)

    train_loss, val_loss = [], []

    best_val_loss = np.inf
    best_val_epoch = 0


    start_time = time.time()
    for epoch in range(1, 1+args.maxep):

        step_losses = []
        tick_ep = time.time()

        model.train()
        for X_mb, Y_mb in dl_train:
            optimizer.zero_grad()

            X_mb_dev = X_mb.cuda()
            Y_mb_dev = Y_mb.cuda()

            if args.aug:
                seed = np.random.choice(10000000, 1)[0]
                torch.manual_seed(seed=seed)
                X_mb_dev = inp_transformer(X_mb_dev)
                torch.manual_seed(seed=seed)
                Y_mb_dev = gt_transformer(Y_mb_dev)

            pred = model(X_mb_dev)
            loss = criterion(pred, Y_mb_dev)
            loss.backward()
            optimizer.step()
            step_losses.append(loss.detach().cpu().numpy())

            #break

        if rank != world_size-1: continue

        ep_time = time.time() - tick_ep
        logging.info(f'\nEpoch {epoch}')
        iter_prints = f'[Train] loss: {np.mean(step_losses):.6f}, {step_losses[0]:.6f} => {step_losses[-1]:.6f}, rate: {ep_time:.2f}s/ep'
        logging.info(iter_prints)


        train_loss.append(np.mean(step_losses))

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            step_losses_val = []
            tick_ep = time.time()

            for X_mb, Y_mb in dl_valid:
                X_mb_dev = X_mb.cuda()
                Y_mb_dev = Y_mb.cuda()

                pred = model(X_mb_dev)
                loss = criterion(pred, Y_mb_dev)

                step_losses_val.append(loss.cpu().numpy())

                #break
                
        ep_time = time.time() - tick_ep

        iter_prints = f'[Val] loss:   {np.mean(step_losses_val):.6f}, {step_losses_val[0]:.6f} => {step_losses_val[-1]:.6f}, rate: {ep_time:.2f}s/ep'
        logging.info(iter_prints)

        val_loss.append(np.mean(step_losses_val))

        #Save the best model 
        if np.mean(step_losses_val) < best_val_loss:
            best_val_loss = np.mean(step_losses_val)
            best_val_epoch = epoch

            mdl_fname = f"{odir}/best_model.pth"
            torch.save({
                'model_state_dict': deepcopy(model.module.state_dict())
            }, mdl_fname)
        

        if epoch % 10 == 0:
            idx = np.random.randint(1, pred.shape[0])
            plt.figure(figsize=(10,10))
            plt.imshow(pred.squeeze().cpu().numpy()[idx][128], cmap='gray')
            plt.savefig(f'{odir}/results/_{epoch}_prediction.png')
            plt.close()

            plt.figure(figsize=(10,10))
            plt.imshow(Y_mb_dev.squeeze().cpu().numpy()[idx][128], cmap='gray')
            plt.savefig(f'{odir}/results/_{epoch}_ground_truth.png')
            plt.close()

        logging.info(f'Lowest validation loss {best_val_loss:.6f} at epoch {best_val_epoch}')

        plt.figure(figsize=(12,8))
        plt.title("Training Progress")
        plt.plot(train_loss[10:], label="Training Loss")
        plt.plot(val_loss[10:], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{odir}/results/__training.png')
        plt.close()


if __name__ == "__main__":


    #path to train data
    train_data = '...'
    #path to val data
    val_data = '...'
    

    parser = argparse.ArgumentParser(description='CBCT Grand Challenge')
    parser.add_argument('-gpus',   type=str, default="", help='list of visiable GPUs')
    parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
    parser.add_argument('-th5',type=str, default=train_data, help='train data h5 format')
    parser.add_argument('-vh5',type=str, default=val_data, help='val data h5 format')
    parser.add_argument('-mbsz',type=int, default=2, help='minibatch size')
    parser.add_argument('-psz',type=int, default=256, help='patch size')
    parser.add_argument('-aug',type=int, default=0, help='data augmentation')
    parser.add_argument('-maxep',type=int, default=100, help='number of training epochs')
    parser.add_argument('-lr',type=float, default=1e-4, help='learning rate')
    parser.add_argument('-mdl',type=str, default=None, help='path to model')
    parser.add_argument('-verbose',type=int, default=1, help='1:print to terminal; 0: redirect to file')
    parser.add_argument('--local_rank', type=int, help='local rank for DDP')

    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ...' % '\n'.join(unparsed))
        exit(0)

    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    logging.getLogger('matplotlib.font_manager').disabled = True

    main(args)
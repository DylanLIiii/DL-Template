import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from utils import check_gpu_status, get_logger, DistributedLogger
from dataclasses import dataclass
import glob

from torch.utils.tensorboard import SummaryWriter

# DDP setting. Generally not use DP, because DDP always better than DP acorrding to pytorch docs.
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# local_rank parameter should exist trainer config 
# In DDP setting, random seed is very important, Make sure use generator for data loader and _worker_init_fn 
# Only log when local_rank is 0
# Generally we use torchrun as a launcher for DDP
import torch.multiprocessing as mp # there are two ways to init a DDP run, multiprocessing or torchrun
from tqdm import tqdm

try:
    import pretty_errors
except ImportError as e:
    print(f"An error occurred: {e}")
    print("Please install the 'pretty_errors' package using 'pip install pretty_errors'.")

try:
    import wandb 
except ImportError as e:    
    print(f"An error occurred: {e}")
    print("Please install the 'wandb' package using 'pip install wandb'")
    print("If you don't want to use wandb, set 'use_wandb' to False in your trainer configuration file. Then will use tensorboard instead.")

#SECTION - Trainer Class
class Trainer:
    """
    A class for training a model using PyTorch, designed to support both single and distributed GPU training.

    This class encapsulates the training process, including model initialization, loss function setup,
    optimizer configuration, learning rate scheduling, and training loop management. It supports
    training on single GPUs as well as multiple GPUs using Distributed Data Parallel (DDP).

    Attributes:
        cfg (dict): Configuration dictionary containing hyperparameters and settings.
        model (nn.Module): The model to be trained.
        device (torch.device): The device on which the model will be trained (CPU or GPU).
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        train_loader (DataLoader): The DataLoader for the training dataset.
        test_loader (DataLoader): The DataLoader for the testing dataset.
        start_epoch (int): The starting epoch for training.
        best_acc (float): The best accuracy achieved during training.
        early_stop_counter (int): A counter for early stopping mechanism.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.

    Example:
        # Initialize the model, dataset, and configuration
        model = MyModel()
        dataset = MyDataset()
        cfg = {...} # Configuration dictionary

        # Create a Trainer instance
        trainer = Trainer(cfg, model, dataset)

        # Start training
        trainer.train()
    """
    
    def __init__(self, cfg, model, dataset, optimizer=None, criterion=None, scheduler=None, logger=None, writer=None, stopper=None, local_rank=None):
        """
        Initializes the Trainer class with the given configuration, model, and data loaders.

        Parameters:
            cfg (dict): Configuration dictionary containing hyperparameters and settings.
            model (nn.Module): The model to be trained. The model can not in device, it should be moved to device in this class.
            dataset (list): A list containing the training and testing DataLoader objects.
            optimizer (torch.optim.Optimizer, optional): The optimizer used for training. Defaults to None.
            criterion (torch.nn.Module, optional): The loss function used for training. Defaults to None.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler. Defaults to None.
            logger (Logger, optional): A logger object for logging. Defaults to None.
            writer (SummaryWriter, optional): A SummaryWriter object for logging. Defaults to None.
            stopper (EarlyStopping, optional): An EarlyStopping object for early stopping. Defaults to None.
            local_rank (int, optional): The local rank for DDP training. Defaults to None.
        """
        self.local_rank = local_rank
        self.cfg = cfg
        self.logger = logger
        self.stopper = stopper
        self.writer = writer
        self.is_gpu, self.is_multiple_gpu = check_gpu_status(cfg, self.logger)

        # Make sure device_ids is a string like 0,1,2
        self.logger.info(f"GPU: {self.is_gpu}, Multiple GPUs: {self.is_multiple_gpu}, Device: cuda:{self.cfg.device_ids}")
        
        # Then init training setting , DDP will init here
        self._init_training_setting()

        # init model
        self._init_model(model)

        self._init_criterion(criterion)
        self._init_optimizer(optimizer)
        self._init_scheduler(scheduler)
        
        self._init_generator()
        
        train_sampler = self.sampler(dataset[0])
        self.train_loader = DataLoader(dataset[0], batch_size=cfg.batch_size, pin_memory=True, num_workers=cfg.num_workers, sampler=train_sampler, generator=self.generator)
        self.test_loader = DataLoader(dataset[1], batch_size=cfg.batch_size, pin_memory=True, num_workers=cfg.num_workers, generator=self.generator)
        
        self.logger.info(
            f"Train loader length: {len(self.train_loader)}, Test loader length: {len(self.test_loader)}"
        )

        self.start_epoch = 1
        
        self.logger.info(f"Start epoch: {self.start_epoch}")

        self.scaler = torch.cuda.amp.GradScaler() if self.cfg.amp is not None else None
        self.logger.info(f"AMP: {self.cfg.amp}")
        
        self.writer = writer

        self.logging_now = True if self.local_rank == 0 else False
        
    def _init_model(self, model):
        """
        Initializes the model based on the configuration.

        If multiple GPUs are available, the model is wrapped with DDP and moved to the appropriate device.
        Otherwise, the model is simply moved to the device specified in the configuration.

        Parameters:
            model (nn.Module): The model to be trained.
        """
        if self.is_multiple_gpu: 
            self.model = model
            self.model = self.model.to(self.device)
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)#find_unused_parameters=True#
        else:
            self.model = model
            self.model = model.to(self.device)
            
    def _setup_ddp(self):
        """
        Set up distributed data parallel (DDP) training.
        """
        devices_str = ",".join(map(str, self.cfg.device_ids))
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12453'
        os.environ["CUDA_VISIBLE_DEVICES"] = devices_str
        os.environ['RANK'] = str(self.local_rank)
        word_size = torch.cuda.device_count()
        os.environ['WORLD_SIZE'] = str(word_size)
        
        
    def _init_training_setting(self):
        if self.is_multiple_gpu:
            self.logger.info("Using DDP Training")
            self._setup_ddp()
            torch.distributed.init_process_group(backend='nccl') # For init_method, will use env as default
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda', self.local_rank)
            self.sampler = DistributedSampler
            self.word_size = dist.get_world_size() # The number of used GPUs
        else: 
            self.logger.info("Using Single GPU Training")
            self.local_rank = 0 # Here, local rank is a dummy value for single gpu
            self.sampler = RandomSampler
            self.device = torch.device(
            'cuda:' + str(self.cfg.device_ids[0]) if self.is_gpu else "cpu"
        )
    
    def _init_generator(self):
        local_rank = dist.get_rank() if self.is_multiple_gpu else 0
        g = torch.Generator()
        #NOTE - In this kind of setting, it will keep the same seed for all processes except dataloader
        g.manual_seed(3407 + local_rank)
        self.generator = g

    def _init_criterion(self, criterion):
        self.criterion = criterion
        if self.criterion is None:
            if self.cfg.Criterion == "cross_entropy":
                self.logger.info("Using default CrossEntropyLoss criterion")
                self.criterion = nn.CrossEntropyLoss()
            else:
                raise NotImplementedError(f"Criterion {self.cfg.Criterion} not implemented")
        else: 
            self.logger.info(f"Using criterion {self.criterion.__class__.__name__}")
            

    def _init_optimizer(self, optimizer):
        self.optimizer = optimizer
        if self.optimizer is None:
            if self.cfg.optimizer == "adam":
                self.logger.info("Using Adam optimizer")
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.cfg.lr,
                    weight_decay=self.cfg.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {self.cfg.optimizer} not implemented")
        else:
            self.logger.info(f"Using optimizer {self.optimizer.__class__.__name__}")

    def _init_scheduler(self, scheduler):
        self.scheduler = scheduler
        if self.scheduler is None:
            if self.cfg.lr_decay == "cosine":
                self.logger.info("Using CosineAnnealingLR scheduler")
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs, eta_min=0)
            else:
                raise NotImplementedError(f"Learning rate decay scheduler {self.cfg.lr_decay} not implemented")
        else:
            self.logger.info(f"Using scheduler {self.scheduler.__class__.__name__}")

    def train_one_step(self, data, target):
        if self.cfg.amp is not None:
            with torch.cuda.amp.autocast():
                self.optimizer.zero_grad()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.zero_grad()
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        #FIXME - Set epoch for DistributedSampler
        if self.is_multiple_gpu:
            self.train_loader.sampler.set_epoch(epoch)
        pbar = tqdm(self.train_loader, disable= not self.logging_now)
        for data, target in pbar:
            loss = self.train_one_step(data, target)
            total_loss += loss
            pbar.set_description(f"Train Epoch: {epoch}, Loss: {loss:.4f}")

        return total_loss / len(self.train_loader)

    def test_one_epoch(self):
        self.model.eval()
        # if use DDP. make sure use tensor on the same device 
        # create on device instead of creating on cpu then move to target device like .cuda()
        total_loss = 0.0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                pred = output.argmax(dim=1)
                total_loss += loss.item()
        #reduce all data from multi-process into a single process
        avg_loss = total_loss / len(self.test_loader)
            
        return avg_loss

    def resume(self, checkpoint_path=None):
        """
        Resume training from a checkpoint.

        Parameters:
        - checkpoint_path (str): Path to the checkpoint file.
        """
        if checkpoint_path is None:
            # load from last and best checkpoint
            checkpoint_path = os.path.join(
                self.cfg.checkpoint_dir, f"{self.cfg.model_name}_best_model.pth"
            )

        checkpoint = torch.load(checkpoint_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_acc = checkpoint["best_acc"]

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.logger.info(
            f"Resuming training from epoch {self.start_epoch} with best accuracy {self.best_acc:.2f}"
        )
    def save_best_model(self, epoch, best_acc):
        if self.local_rank == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_acc": best_acc,
            }
            checkpoint_path = os.path.join(
                self.cfg.checkpoint_dir, f"{self.cfg.model_name}_best_model.pth"
            )
            # check if it exits
            if not os.path.exists(self.cfg.checkpoint_dir):
                os.mkdir(self.cfg.checkpoint_dir)
            if self.local_rank == 0:
                torch.save(checkpoint, checkpoint_path)
                self.logger.info(f"Best model saved to {checkpoint_path}")
    
    def cleanup(self):
        if self.is_multiple_gpu:
            dist.destroy_process_group() ### 最后摧毁进程，和 init_process_group 相对
        else:
            torch.cuda.empty_cache()

    def train(self):
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            self.scheduler.step()
            test_loss = self.test_one_epoch()
            
            if self.local_rank == 0:
               self.stopper(test_loss)
            
            self.stopper.check_stop()
            
            # for logging, should reduce all into 0 local rank 
            dist.reduce(torch.tensor(train_loss, device=self.device), 0, op=dist.ReduceOp.SUM)
            dist.reduce(torch.tensor(test_loss, device=self.device), 0, op=dist.ReduceOp.SUM)
            
            self.writer.add_scalar('train_loss', train_loss, epoch)
            self.writer.add_scalar('test_loss', test_loss, epoch)
            
            self.logger.info(
                f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
            )
            if self.stopper.is_best_now():
                self.save_best_model(epoch, test_loss)
            
            if self.stopper.early_stop:
                self.logger.info("Early stopping")
                break

        self.cleanup()                
#!SECTION

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, local_rank=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.local_rank = local_rank

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def check_stop(self):
        stop = torch.tensor(int(self.early_stop), dtype=torch.int, device='cuda')
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        self.early_stop = bool(stop.item() > 0)
        if self.local_rank == 0:
            print(f"Early stop flag after synchronization: {self.early_stop}")
            
    def is_best_now(self):
        if self.counter == 0:
            return True
        else:
            return False
            
#SECTION - Test Trainer Code            
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, input_dim, num_classes):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = torch.randn(self.input_dim)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y
    
def test_trainer_function(local_rank):
    # test trainer 
    @dataclass
    class Config:
        amp: bool = True
        Criterion: str = 'cross_entropy'
        lr: float = 0.001
        weight_decay: float = 0.0
        optimizer: str = 'adam'
        lr_decay: str = 'cosine'
        epochs: int = 50
        device_ids = [0, 1, 2, 3]
        checkpoint_dir: str = 'checkpoints'
        model_name: str = 'test_model'
        early_stop_patience: int = 5
        single_gpu: bool = False
        checkpoint_dir: str = 'checkpoints'
        batch_size: int = 32
        num_workers: int = 4
    cfg = Config()

    model = nn.Linear(10, 2)
    # Create synthetic datasets
    train_dataset = SyntheticDataset(num_samples=10000, input_dim=10, num_classes=2)
    test_dataset = SyntheticDataset(num_samples=200, input_dim=10, num_classes=2)

    # Instantiate the Trainer with the model, data loaders, and configuration
    logger = DistributedLogger(name="Trainer", local_rank=local_rank)
    stopper = EarlyStopping(patience=cfg.early_stop_patience, local_rank=local_rank)
    # if you do not wanna use wandb, set mode = 'disabled'
    writer = DistrubutedWriter(log_dir="test_log_dir", local_rank=local_rank)
    trainer = Trainer(cfg=cfg, model=model, dataset=(train_dataset, test_dataset), local_rank=local_rank, logger=logger, stopper=stopper, writer=writer)

    # Run the training
    trainer.train()
    writer.close()
    
class DistrubutedWriter():
    def __init__(self, log_dir='None', comment="", purge_step=None, max_queue=10, flush_secs=120, filename_suffix="", local_rank=0):
        self.log_dir = log_dir
        self.comment = comment
        self.local_rank = local_rank
        self._init_wandb(mode='online')
        self._init_tensorboard()

    # Only override the add_scalar method. Because we use it here.
    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if self.local_rank == 0:
            self.tensorboard_writer.add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime)
            if hasattr(self, 'wandb_run'): 
                self.wandb_run.log({f'{tag}': scalar_value, 'step': global_step})

            
    def _init_wandb(self, mode):
        # must call either wandb.init or wandb.tensorboard.patch before calling tf.summary.create_file_writer or constructing a SummaryWriter via torch.utils.tensorboard.SummaryWriter
        # can accept a config parameter
        if self.local_rank == 0:
            wandb.tensorboard.patch(root_logdir=self.log_dir, pytorch=True)
            self.wandb_run = wandb.init(project='test', mode=mode, sync_tensorboard=True)
    
    def _init_tensorboard(self):
        if self.local_rank == 0:
            self.tensorboard_writer = SummaryWriter(log_dir=self.log_dir)
    def close(self):
        if hasattr(self, 'wandb_run'):
            #wandb.save(glob.glob(f"{self.log_dir}/*.pt.trace.json")[0], base_path=f"{self.log_dir}")
            wandb.finish()
        self.tensorboard_writer.close()

    
#!SECTION
if __name__ == "__main__":
    import time
    time_start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    #writer.add_scalar("test", 0.5, 0)
    #writer.close()
    
    mp.spawn(test_trainer_function, args=(), nprocs=4)
    time_elapsed = time.time() - time_start
    
    #writer.close()
    print(f"\ntime elapsed: {time_elapsed:.2f} seconds")
    
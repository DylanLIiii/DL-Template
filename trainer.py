import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from utils import check_gpu_status, get_logger
from dataclasses import dataclass

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
from tqdm import tqdm

logger = get_logger(__name__)

try:
    import wandb 
except ImportError as e:    
    logger.warning(f"An error occurred: {e}")
    logger.warning("Please install the 'wandb' package using 'pip install wandb'")
    logger.warning("If you don't want to use wandb, set 'use_wandb' to False in your trainer configuration file. Then will use tensorboard instead.")
    
class Trainer:
    """
    A class for training a model using PyTorch.

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

    Methods:
        __init__(self, cfg, model, Datasetloader, optimizer=None, criterion=None, scheduler=None):
            Initializes the Trainer class with the given configuration, model, and data loaders.
        _init_criterion(self):
            Initializes the loss function based on the configuration.
        _init_optimizer(self):
            Initializes the optimizer based on the configuration.
        _init_scheduler(self):
            Initializes the learning rate scheduler based on the configuration.
        train_one_step(self, data, target):
            Performs one training step with the given data and target.
        train_one_epoch(self, epoch):
            Trains the model for one epoch.
        test_one_epoch(self):
            Tests the model for one epoch.
        save_best_model(self, epoch, best_acc):
            Saves the best model based on accuracy.
        resume(self, checkpoint_path=None):
            Resumes training from a checkpoint.
        early_stop(self, test_acc):
            Implements early stopping based on test accuracy.
        train(self):
            Trains the model for the specified number of epochs.
    """
    
    def __init__(self, cfg, model, dataset, optimizer=None, criterion=None, scheduler=None, writer=None):
        """
        Initializes the Trainer class with the given configuration, model, and data loaders.

        Parameters:
            cfg (dict): Configuration dictionary containing hyperparameters and settings.
            model (nn.Module): The model to be trained. The model can not in device, it should be moved to device in this class.
            dataloader (list): A list containing the training and testing DataLoader objects.
            optimizer (torch.optim.Optimizer, optional): The optimizer used for training. Defaults to None.
            criterion (torch.nn.Module, optional): The loss function used for training. Defaults to None.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler. Defaults to None.
            writer (SummaryWriter, optional): A SummaryWriter object for logging. Defaults to None.
            #NOTE - For writer, generally we use tensorboard, can use wandb
        """
        
        self.cfg = cfg
        self.is_gpu, self.is_multiple_gpu = check_gpu_status(cfg)
        logger.info(f"GPU: {self.is_gpu}, Multiple GPUs: {self.is_multiple_gpu}, Device: {self.cfg.device_ids}")
        
        # Firstly init device between cuda and cpu
        # The first device_ids determine the main device for single gpu
        self.device = torch.device(
            self.cfg.device_ids[0] if self.is_gpu else "cpu"
        )
        # init model first
        self.model = model
        self.model = self.model.to(self.device)
        
        # Then init training setting , DDP will init here
        self._init_training_setting()

        self.optimizer = optimizer
        self.scheduler = criterion
        self.criterion = scheduler
        train_sampler = self.sampler(dataset[0])
        self.train_loader = DataLoader(dataset[0], batch_size=cfg.batch_size, pin_memory=True, num_workers=cfg.num_workers, sampler=train_sampler, generator=self.generator)
        self.test_loader = DataLoader(dataset[1], batch_size=cfg.batch_size, pin_memory=True, num_workers=cfg.num_workers, generator=self.generator)
        
        logger.info(
            f"Train loader length: {len(self.train_loader)}, Test loader length: {len(self.test_loader)}"
        )

        self.start_epoch = 1
        
        logger.info(f"Start epoch: {self.start_epoch}")
        self.best_acc = 0.0
        self.early_stop_counter = 0

        self.scaler = torch.cuda.amp.GradScaler() if self.cfg.amp is not None else None
        logger.info(f"AMP: {self.cfg.amp}")
        
        self.writer = writer
        
        self._init_criterion()
        self._init_optimizer()
        self._init_scheduler()
        
        self.logging_now = True if self.local_rank == 0 else False
        
    def _init_training_setting(self):
        if self.is_multiple_gpu:
            logger.info("Using DDP Training")
            torch.distributed.init_process_group(backend='nccl') # For init_method, will use env as default
            self.local_rank = torch.distributed.get_rank()
            self.word_size = torch.distributed.get_world_size() # The number of used GPUs
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda', self.local_rank)
            self.sampler = DistributedSampler
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        else: 
            logger.info("Using Single GPU Training")
            self.local_rank = 0 # Here, local rank is a dummy value for single gpu
            self.sampler = RandomSampler
    
    def _get_generator(self):
        local_rank = dist.get_rank() if self.is_multiple_gpu else 0
        g = torch.Generator()
        #NOTE - In this kind of setting, it will keep the same seed for all processes except dataloader
        g.manual_seed(3407 + local_rank)
        self.generator = g

    def _init_criterion(self):
        if self.criterion is None:
            if self.cfg.Criterion == "cross_entropy":
                logger.info("Using default CrossEntropyLoss criterion")
                self.criterion = nn.CrossEntropyLoss()
            else:
                raise NotImplementedError(f"Criterion {self.cfg.Criterion} not implemented")
        else: 
            logger.info(f"Using criterion {self.criterion.__class__.__name__}")
            

    def _init_optimizer(self):
        if self.optimizer is None:
            if self.cfg.optimizer == "adam":
                logger.info("Using Adam optimizer")
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.cfg.lr,
                    weight_decay=self.cfg.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {self.cfg.optimizer} not implemented")
        else:
            logger.info(f"Using optimizer {self.optimizer.__class__.__name__}")

    def _init_scheduler(self):
        if self.scheduler is None:
            if self.cfg.lr_decay == "cosine":
                logger.info("Using CosineAnnealingLR scheduler")
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs, eta_min=0)
            else:
                raise NotImplementedError(f"Learning rate decay scheduler {self.cfg.lr_decay} not implemented")
        else:
            logger.info(f"Using scheduler {self.scheduler.__class__.__name__}")

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
        total_loss = 0.0
        total_correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                pred = output.argmax(dim=1)
                total_loss += loss.item()
                total_correct += pred.eq(target).sum().item()
                
        if self.logging_now: # Only log when local_rank is 0
            avg_loss = total_loss / len(self.test_loader)
            avg_accuracy = total_correct / len(self.test_loader.dataset)
        return avg_loss, avg_accuracy

    def save_best_model(self, epoch, best_acc):
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
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Best model saved to {checkpoint_path}")

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

        logger.info(
            f"Resuming training from epoch {self.start_epoch} with best accuracy {self.best_acc:.2f}"
        )

    def early_stop(self, test_acc):
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.cfg.early_stop_patience:
                logger.info(
                    f"Early stopping triggered. Best accuracy: {self.best_acc:.2f}"
                )
                return True
        return False

    def train(self):
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            self.scheduler.step()
            test_loss, test_acc = self.test_one_epoch()

            logger.info(
                f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}"
            )

            if test_acc > self.best_acc:
                self.save_best_model(epoch, test_acc)

            if self.early_stop(test_acc):
                break

if __name__ == '__main__':
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    # test trainer 
    print(f'{"-"*10} test Trainer {"-"*10}')
    @dataclass
    class Config:
        amp: bool = True
        Criterion: str = 'cross_entropy'
        lr: float = 0.001
        weight_decay: float = 0.0
        optimizer: str = 'adam'
        lr_decay: str = 'cosine'
        epochs: int = 10
        device_ids = ['cuda:0', 'cuda:1']
        checkpoint_dir: str = 'checkpoints'
        model_name: str = 'test_model'
        early_stop_patience: int = 3
        single_gpu: bool = False
        checkpoint_dir: str = 'checkpoints'
    cfg = Config()
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
    model = nn.Linear(10, 2)
    # Create synthetic datasets
    train_dataset = SyntheticDataset(num_samples=1000, input_dim=10, num_classes=2)
    test_dataset = SyntheticDataset(num_samples=200, input_dim=10, num_classes=2)

    # Instantiate the Trainer with the model, data loaders, and configuration
    trainer = Trainer(cfg=cfg, model=model, dataset=(train_dataset, test_dataset))

    # Run the training
    trainer.train()
    
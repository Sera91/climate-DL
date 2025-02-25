import time
import sys
import pickle

import torch

#------Some useful utilities------

class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    (from the Deep Learning tutorials of DSSC)
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def accuracy_binary_one(prediction, target):
    prediction_class = torch.where(prediction > 0.5, 1.0, 0.0) 
    correct_items = (prediction_class == target)
    acc = correct_items.sum().item() / prediction.shape[0]  
    return acc

def accuracy_binary_one_class1(prediction, target):
    prediction_class = torch.where(prediction > 0.5, 1.0, 0.0)
    correct_items = (prediction_class == target)[target==1.0]
    if correct_items.shape[0] > 0:
        acc_class1 = correct_items.sum().item() / correct_items.shape[0]
        return acc_class1
    else:
        return 0.0

def accuracy_binary_two(prediction, target):
    prediction_class = torch.argmax(prediction, dim=-1).squeeze()
    correct_items = (prediction_class == target)
    acc = correct_items.sum().item() / prediction.shape[0]  
    return acc

def accuracy_binary_two_class1(prediction, target):
    prediction_class = torch.argmax(prediction, dim=-1).squeeze()
    correct_items = (prediction_class == target)[target==1.0]
    if correct_items.shape[0] > 0:
        acc_class1 = correct_items.sum().item() / correct_items.shape[0]
        return acc_class1
    else:
        return 0.0

def weighted_mse_loss(input_batch, target_batch, weights):
    return (weights * (input_batch - target_batch) ** 2).sum() / weights.sum()

def load_encoder_checkpoint(model, checkpoint, log_path, log_file, accelerator, net_names, fine_tuning=True):
    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write("\nLoading encoder parameters.") 
    checkpoint = torch.load(checkpoint)
    state_dict = checkpoint["parameters"]
    for name, param in state_dict.items():
        for net_name in net_names:
            if net_name in name:
                if accelerator is None or accelerator.is_main_process:
                    with open(log_path+log_file, 'a') as f:
                        f.write(f"\nLoading parameters '{name}'")
                param = param.data
                if name.startswith("module"):
                    name = name.partition("module.")[2]
                try:
                    model.state_dict()[name].copy_(param)
                except:
                     if accelerator is None or accelerator.is_main_process:
                        with open(log_path+log_file, 'a') as f:
                            f.write(f"\nParam {name} was not loaded..")
    if not fine_tuning:
        for net_name in net_names:
            [param.requires_grad_(False) for name, param in model.named_parameters() if net_name in name]
    return model


def check_freezed_layers(model, log_path, log_file, accelerator):
    for name, param in model.named_parameters():
        n_param = param.numel() 
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nLayer {name} requires_grad = {param.requires_grad} and has {n_param} parameters") 


class Trainer(object):

    def _train_epoch_ae(self, epoch, model, dataloader, optimizer, loss_fn, accelerator, args, lr_scheduler):
        loss_meter = AverageMeter()     
        start = time.time()
        step = 0
        for X in dataloader:
            optimizer.zero_grad()
            X_pred = model(X)
            loss = loss_fn(X_pred, X)
            accelerator.backward(loss)
            optimizer.step()
            loss_meter.update(val=loss.item(), n=X.shape[0])
            accelerator.log({'epoch':epoch, 'loss iteration': loss_meter.val, 'loss avg': loss_meter.avg, 'lr': lr_scheduler.get_last_lr()[0], 'step':step})
            #if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.000001:
            #lr_scheduler.step() 
            if accelerator.is_main_process and step % 50000 == 0:
                    checkpoint_dict = {
                        "parameters": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        }
                    torch.save(checkpoint_dict, args.output_path+f"checkpoint_{epoch}_tmp.pth")
            step += 1  
        end = time.time()
        accelerator.log({'loss epoch': loss_meter.avg})
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. ")

    def _train_epoch_cl(self, epoch, model, dataloader, optimizer, loss_fn, accelerator, args, lr_scheduler, alpha=0.9, gamma=2):
        loss_meter = AverageMeter()
        performance_meter = AverageMeter()
        acc_class1_meter = AverageMeter()
        start = time.time()
        step = 0
        device = 'cuda' if accelerator is None else accelerator.device
        for X, data in dataloader:
            optimizer.zero_grad()
            y_pred, y = model(X, data, device)
            #loss = loss_fn(y_pred, y)
            loss = loss_fn(y_pred, y, alpha, gamma, reduction='mean')
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            loss_meter.update(val=loss.item(), n=X.shape[0])    
            performance = accuracy_binary_one(y_pred, y)
            acc_class1 = accuracy_binary_one_class1(y_pred, y)
            performance_meter.update(val=performance, n=X.shape[0])
            acc_class1_meter.update(val=acc_class1, n=X.shape[0])
            #if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.000001:
            accelerator.log({'epoch':epoch, 'loss iteration': loss_meter.val, 'accuracy iteration': performance_meter.val, 'loss avg': loss_meter.avg,
                'accuracy avg': performance_meter.avg, 'accuracy class1 avg': acc_class1_meter.avg, 'lr': lr_scheduler.get_last_lr()[0], 'step':step})
            #lr_scheduler.step()
            step += 1
            if accelerator.is_main_process:
                if step % 5000 == 0:
                    checkpoint_dict = {
                        "parameters": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        }
                    torch.save(checkpoint_dict, args.output_path+f"checkpoint_{epoch}_tmp.pth")
            #print("OK")
            #sys.exit()
        end = time.time()
        accelerator.log({'loss epoch': loss_meter.avg, 'accuracy epoch': performance_meter.avg, 'accuracy class1 epoch': acc_class1_meter.avg})
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}; "+
                    f"performance: {performance_meter.avg:.4f}.")

    def _train_epoch_reg(self, epoch, model, dataloader, optimizer, loss_fn, accelerator, args, lr_scheduler):
        loss_meter = AverageMeter()
        start = time.time()
        step = 0 
        #t0 = time.time()
        device = 'cuda' if accelerator is None else accelerator.device
        for X, data in dataloader:
            optimizer.zero_grad()
            y_pred, y = model(X, data, device)
            loss = loss_fn(y_pred, y)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            loss_meter.update(val=loss.item(), n=X.shape[0])    
            #grad_max = torch.max(torch.abs(torch.cat([param.grad.view(-1) for param in model.parameters()]))).item()
            accelerator.log({'epoch':epoch, 'loss iteration': loss_meter.val, 'loss avg': loss_meter.avg, 'step':step})
                #'lr': lr_scheduler.get_last_lr()[0]}) #, 'grad_max':grad_max})
            #if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.000001:
            #lr_scheduler.step()
            step += 1
            if accelerator.is_main_process:
                if step % 5000 == 0:
                    checkpoint_dict = {
                        "parameters": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        }
                    torch.save(checkpoint_dict, args.output_path+f"checkpoint_{epoch}_tmp.pth")
            #t0 = time.time()
            #print("OK")
            #sys.exit()
        end = time.time()
        accelerator.log({'loss epoch': loss_meter.avg})
        if accelerator.is_main_process:
            with open(args.output_path+args.log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. ")
    
    def train(self, model, dataloader, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=0):
        model.train()
        epoch_type = 'reg' if 'reg' in args.model_type else args.model_type
        train_epoch = getattr(self, f"_train_epoch_{epoch_type}")
        for epoch in range(epoch_start, epoch_start+args.epochs):
            if accelerator.is_main_process:
                with open(args.output_path+args.log_file, 'a') as f:
                    f.write(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}")
            train_epoch(epoch, model, dataloader, optimizer, loss_fn, accelerator, args, lr_scheduler)
            if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.000001:
                lr_scheduler.step()
            if accelerator.is_main_process:
                checkpoint_dict = {
                    "parameters": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    }
                torch.save(checkpoint_dict, args.output_path+f"checkpoint_{epoch}.pth")


class Get_encoder(object):

    def get_encoding(self, model, dataloader, accelerator, args, space_dim=495, time_dim=130727):
        model.eval()
        encodings_array = torch.zeros((space_dim+1, time_dim+1, 128), dtype=torch.float32)
        step = 0
        with torch.no_grad():
            for X, idxs in dataloader: # idxs.shape = (batch_dim, 2)
                encodings = model(X.cuda())
                encodings_array[idxs[:,0], idxs[:,1], :] = encodings.cpu()
                if step == 10:
                    break
                #for i, e in enumerate(encodings):
                #    s = idxs[i,0].item(); t = idxs[i,1].item()
                #    encodings_array[s,t,:] = e.cpu().numpy()
                if step % 25000 == 0:
                    with open(args.output_path+args.log_file, 'a') as f:
                        f.write(f"\nStep {step} done.")
                step += 1
            with open(args.output_path+"encodings_array.pkl", 'wb') as f:
                pickle.dump(encodings_array, f)

class Tester(object):

    def test(self, model_cl, model_reg, dataloader, G_test, args, accelerator=None):
        model_cl.eval()
        model_reg.eval()
        step = 0
        device = 'cuda' if accelerator is None else accelerator.device
        with torch.no_grad():    
            for X, data in dataloader:
                X = X.cuda()
                model_cl(X, data, G_test, device)
                model_reg(X, data, G_test, device)
                
                if step % 100 == 0:
                    with open(args.output_path+args.log_file, 'a') as f:
                        f.write(f"\nStep {step} done.") # {G_test.pr_reg.isnan().sum() / (2334 * 9528)}, {G_test.pr_cl.isnan().sum() / (2334 * 9528)}")
                step += 1 
        G_test["pr"] = G_test.pr_cl * G_test.pr_reg
        
        return

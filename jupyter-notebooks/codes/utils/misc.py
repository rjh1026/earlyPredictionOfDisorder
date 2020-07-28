import torch


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarr):
    if type(ndarr).__module__ == 'numpy':
        return torch.from_numpy(ndarr)
    elif not torch.is_tensor(ndarr):
        raise ValueError("Cannot convert {} to tensor".format(type(ndarr)))
    return ndarr


def save_checkpoint(md_kwargs, ds_kwargs, model, optimizer, loss, tloss, tacc, eval_met, epoch, path, scheduler=None):
    
    model.train() # to save batchnorm, dropout, etc..
    filepath = path + '_epoch_' + str(epoch) + '.pth.tar'

    torch.save({
        'meta': {
            'train_loss': loss,
            #'train_acc': acc,
            'test_loss': tloss,
            'test_acc': tacc,
            'eval_met': eval_met,
            'cur_epoch': epoch
        },
        'md_kwargs': md_kwargs,
        'ds_kwargs': ds_kwargs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler != None else None 
    }, filepath)


def load_checkpoint(epoch, path):
    filepath = path + '_epoch_' + str(epoch) + '.pth.tar'

    return torch.load(filepath)
import time

import torch
import torch.nn.functional as F
from torchvision import transforms

def compute_epoch_loss_autoencoder(model, data_loader, loss_fn, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)
            logits, _ = model(features)
            loss = loss_fn(logits, features, reduction='sum')
            num_examples += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss

def compute_epoch_loss_vae(model, data_loader, loss_fn, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)
            _,_,_,logits = model(features)
            loss = loss_fn(logits, features, reduction='sum')
            num_examples += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss

def train_vae(num_epochs, model, optimizer, device, 
                         train_loader, test_loader=None, loss_fn=None,
                         logging_interval=100,  skip_epoch_stats=False, reconstruction_term_weight=1, save_model=None):
    
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': [],
                'test_combined_loss_per_epoch': []}
    
    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):

            features = features.to(device)

            encoded, z_mean, z_log_var, decoded = model(features)

            kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), axis=1)
            batchsize = kl_div.size(0)
            kl_div = kl_div.mean()

            tmp_loss = loss_fn(decoded, features, reduction='none')
            #print(tmp_loss.shape)
            tmp_loss = tmp_loss.view(batchsize, -1).sum(axis=1)#aixs ma byc 1
            #print(tmp_loss.shape)
            tmp_loss = tmp_loss.mean()
           # print(tmp_loss.shape)

            loss=reconstruction_term_weight*tmp_loss+kl_div
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(tmp_loss.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))

        if not skip_epoch_stats:
            model.eval()
            
            with torch.set_grad_enabled(False):  # save memory during inference
                
                train_loss = compute_epoch_loss_vae(
                    model, train_loader, loss_fn, device)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                      epoch+1, num_epochs, train_loss))
                log_dict['train_combined_loss_per_epoch'].append(train_loss.item())

        if test_loader is not None and not skip_epoch_stats:
            model.eval()
            
            with torch.set_grad_enabled(False):  # save memory during inference
                
                test_loss = compute_epoch_loss_vae(
                    model, test_loader, loss_fn, device)
                print('Test***Epoch: %03d/%03d | Loss: %.3f' % (
                      epoch+1, num_epochs, test_loss))
                log_dict['test_combined_loss_per_epoch'].append(test_loss.item())

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    return log_dict

def train_autoencoder(num_epochs, model, optimizer, device,  train_loader, skip_epoch_stats=False, loss_fn=None, logging_interval=1, save_model=None ,test_loader=None):
    
    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': [],
                'test_loss_per_epoch': []}
    
    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):


            features = features.to(device)
            logits, encoded  = model(features)
            loss = loss_fn(logits, features)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            log_dict['train_loss_per_batch'].append(loss.item())
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))

        if not skip_epoch_stats:
            model.eval()
            
            with torch.set_grad_enabled(False):  # save memory during inference
                
                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn, device)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                      epoch+1, num_epochs, train_loss))
                log_dict['train_loss_per_epoch'].append(train_loss.item())

        if test_loader is not None and not skip_epoch_stats:
            model.eval()
            
            with torch.set_grad_enabled(False):  # save memory during inference
                
                test_loss = compute_epoch_loss_autoencoder(
                    model, test_loader, loss_fn, device)
                print('Test***Epoch: %03d/%03d | Loss: %.3f' % (
                      epoch+1, num_epochs, test_loss))
                log_dict['test_loss_per_epoch'].append(test_loss.item())



        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    return log_dict
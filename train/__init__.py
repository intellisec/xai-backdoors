# System
import os

# Libs
import torch

# Own sources

def test_model(model, test_loader, at_a_time=1000):
    device = os.getenv('CUDADEVICE')
    if type(test_loader) is tuple:
        xa, ya = test_loader
        xs, ys = xa.split(at_a_time), ya.split(at_a_time)
        correct = 0
        for x, y in zip(xs, ys):
            with torch.no_grad():
                out = model(x.to(device))
                pred = out.argmax(-1)
                correct += (y.to(device) == pred).count_nonzero().cpu().detach().item()
                #print(correct, x.shape, y.shape, '\n', pred, y)
        #print('accuracy:', correct / x.shape[0])
        #print(correct, 'out of', x.shape[0])
        return correct / xa.shape[0]

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device))
            pred = out.argmax(-1)
            correct += (y.to(device) == pred).count_nonzero().cpu()
    #print('accuracy:', correct / len(test_loader.dataset))
    #print(correct, 'out of', len(test_loader.dataset))
    return correct / len(test_loader.dataset)

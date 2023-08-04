import sys
sys.path.insert(0,'..')
import torch
from torch import nn
from tqdm import tqdm
from train import train, Losses
import priors
import encoders
import positional_encodings
import utils
import bar_distribution
# from samlib.utils import chunker

def pretrain_and_eval(extra_prior_kwargs_dict_eval,*args, **kwargs):
    r = train(*args, **kwargs)
    model = r[-1]
    kwargs['extra_prior_kwargs_dict'] = extra_prior_kwargs_dict_eval
#    acc = get_acc(model, -1, device='cuda:0', **kwargs).cpu()
    acc = get_acc(model, -1, device='cpu', **kwargs).cpu()
    model.to('cpu')
    return r, acc

mykwargs = \
{
 'bptt': 5*5+1,
'nlayers': 6,
 'dropout': 0.0, 'steps_per_epoch': 100,
 'batch_size': 100}

mnist_jobs_5shot_pi_prior_search = [
    pretrain_and_eval({
                        'num_features': 28 * 28, 'fuse_x_y': False, 'num_outputs': 5,
                        'translations': False, 'jonas_style': True
                      },
                      priors.stroke.DataLoader, Losses.ce, enc, emsize=emsize,
                      nhead=nhead, warmup_epochs=warmup_epochs, nhid=nhid,
                      y_encoder_generator=encoders.get_Canonical(5), lr=lr, epochs=epochs,
                      single_eval_pos_gen=mykwargs['bptt']-1,
                      extra_prior_kwargs_dict={
                          'num_features': 28*28, 'fuse_x_y': False, 'num_outputs':5,
                          'only_train_for_last_idx': True,
                          'min_max_strokes': (1,max_strokes),
                          'min_max_len': (min_len, max_len),
                          'min_max_width': (min_width, max_width),
                          'max_offset': max_offset,
                          'max_target_offset': max_target_offset},
                      **mykwargs)
    #max_strokes = 3
    #min_len, max_len = (5/28, 20/28)
    #min_width, max width = (1/28, 4/28) , max_offset=4/28 , maxtargetoffset=2/28
    for max_strokes, min_len, max_len, min_width, max_width, max_offset, max_target_offset in \
    #    random_hypers
         [(3, 5/28, 20/28, 1/28, 4/28, 4/28, 2/28)]
    for enc in [encoders.Linear]
    for emsize in [1024]
    for nhead in [4]
    for nhid in [emsize*2]
    for warmup_epochs in [5]
    for lr in [.00001]
    for epochs in [128,1024]
    for _ in range(1)
]

@torch.inference_mode()
def get_acc(finetuned_model, eval_pos, device='cpu', steps=100, train_mode=False, **mykwargs):
    finetuned_model.to(device)
    finetuned_model.eval()

    t_dl = priors.omniglot.DataLoader(steps,
                                      batch_size=1000, seq_len=mykwargs['bptt'],
                                      train=train_mode,
                                      **mykwargs['extra_prior_kwargs_dict'])

    ps = []
    ys = []
    for x, y in tqdm(t_dl):
        p = finetuned_model(tuple(e.to(device) for e in x), single_eval_pos=eval_pos)
        ps.append(p)
        ys.append(y)

    ps = torch.cat(ps, 1)
    ys = torch.cat(ys, 1)

    def acc(ps, ys):
        return (ps.argmax(-1) == ys.to(ps.device)).float().mean()

    a = acc(ps[eval_pos], ys[eval_pos]).cpu()
    print(a.item())
    return a


def train_and_eval(*args, **kwargs):
    r = train(*args, **kwargs)
    model = r[-1]
    acc = get_acc(model, -1, device='cuda:0', **kwargs).cpu()
    model.to('cpu')
    return [acc]

def pretrain_and_eval(extra_prior_kwargs_dict_eval,*args, **kwargs):
    r = train(*args, **kwargs)
    model = r[-1]
    kwargs['extra_prior_kwargs_dict'] = extra_prior_kwargs_dict_eval
    acc = get_acc(model, -1, device='cuda:0', **kwargs).cpu()
    model.to('cpu')
    return r, acc

emsize = 1024
# mnist_jobs_5shot_pi[20].result()[-1].state_dict()
mykwargs = \
    {'bptt': 5 * 5 + 1,
     'nlayers': 6,
     'nhead': 4, 'emsize': emsize,
     'encoder_generator': encoders.Linear, 'nhid': emsize * 2}
results = train_and_eval(priors.omniglot.DataLoader,
                         Losses.ce, y_encoder_generator=encoders.get_Canonical(5),
                         load_weights_from_this_state_dict=mnist_jobs_5shot_pi_prior_search[67][0][-1].state_dict(), 
                         epochs=32, lr=.00001, dropout=dropout,
                         single_eval_pos_gen=mykwargs['bptt'] - 1,
                         extra_prior_kwargs_dict={
                             'num_features': 28 * 28,
                             'fuse_x_y': False,
                             'num_outputs': 5,
                             'translations': True, 'jonas_style': True},
                         batch_size=100,
                         steps_per_epoch=200,
                         **mykwargs)

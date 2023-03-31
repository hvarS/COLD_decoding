import torch 
import numpy as np 
import torch.nn.functional as F

from util import (
    one_hot, 
    initialize, 
    get_text_from_logits,
    top_k_filter_3d, 
    soft_forward, 
    soft_forward_xyz,
    soft_backward,
    soft_nll, 
    batch_log_bleulosscnn_ae,
    decode_with_model_topk,
    post_process
)
# TODO : Encode, Tensorify and One Hot the Input text
def eto(text, tokenizer, device):
    
    text_ = tokenizer.encode(text)
    text_t = torch.tensor(text_, device=device, dtype=torch.long)
    text_onehot = one_hot(text_t, dimension = len(tokenizer))

    return text_, text_t, text_onehot


def decode(model, model_back, tokenizer, device, x="", z="", constraints=None, args=None, zz=None):
    '''
    x: left context   (prompt in lexical lexical task)
    z: optimization target  (original ending in counterfactual task)
    constraints: (constraint set in lexical constrained task)
    '''
    
    x_, x_t, x_onehot = eto(x,tokenizer,device)     
    # x_, x_t : [len_x]
    # x_onehot : [len_x, vocab_size]

    # repeat batch_size times
    x_t = x_t.unsqueeze(0).repeat(args.batch_size, 1)
    x_onehot = x_onehot.repeat(args.batch_size, 1, 1)
    # x_t: [batch_size,len_x]
    # x_onehot: [batch_size, len_x, vocab_size]

    z_mask = None

    if 'lexical' in args.mode:
        length = args.length

        z_, z_t, z_onehot = eto(z[1:],tokenizer,device)  # delete the "." token we appended before
        # z is the constraints seperated by ' ' 
        # z_, z_t : [len_z]
        # z_onehot : [len_z, vocab_size]

        # repeat batch_size times
        z_t = z_t.unsqueeze(0).repeat(args.batch_size, 1)
        z_onehot = z_onehot.repeat(args.batch_size, 1, 1)
        # z_t: [batch_size,len_z]
        # z_onehot: [batch_size, len_z, vocab_size]
        

        zz_, zz_t, zz_onehot = eto(zz[1:],tokenizer,device)
        zz_t = zz_t.unsqueeze(0).repeat(args.batch_size, 1)
        # zz is the key constraints, same as z in this case 
        # zz_, zz_t : [len_zz]
        # zz_onehot : [len_zz, vocab_size]

        z_mask = np.zeros([len(tokenizer)])   # [vocab_size]
        z_mask[zz_] = 1.
        z_mask = torch.tensor(z_mask, device=device)
        z_mask = z_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)
        # Length here the start length (init)
        # z_mask = [batch_size, length, vocab_size]

        if args.verbose:
            print("x:\t|%s|\nz:\t|%s|\nzz:\t|%s|\nconstraints:\t%s" % (
                tokenizer.decode(x_), tokenizer.decode(z_), tokenizer.decode(zz_), constraints))

    cs_ = None
    cs_onehot = None

    model.eval()

    if args.init_mode == 'random':
        init_logits = initialize(model, x_t, length, args.init_temp, device)
    else:
        init_logits = z_onehot / 0.1
        init_logits = init_logits[:, :length, :]
        if length > init_logits.shape[1]:
            init_logits = torch.cat(
                [init_logits,
                 torch.zeros([args.batch_size, length - init_logits.shape[1], tokenizer.vocab_size], device=device)],
                dim=1)
    
    # Init Logits: [batch_size, length, vocab]
    text, _, _ = get_text_from_logits(init_logits, tokenizer)
    for bi in range(args.batch_size):
        print("[initial]: %s" % (text[bi]))

    # if args.wandb:
    #     wandb.init(
    #         project='args.mode' + str(int(round(time.time() * 1000))),
    #         config=args)

    assert args.prefix_length <= 0  # Otherwise not compatible with batch mode

    if args.prefix_length > 0:
        prefix_logits = torch.nn.Parameter(
            torch.rand(x_onehot.shape[0], args.prefix_length, x_onehot.shape[2], dtype=init_logits.dtype,
                       device=device))
    # y_logits: [batch_size, length, vocab_size]
    y_logits = init_logits
    epsilon = torch.nn.Parameter(torch.zeros_like(y_logits))
    if args.prefix_length > 0:
        optim = torch.optim.Adam([epsilon, prefix_logits], lr=args.stepsize)
    else:
        optim = torch.optim.Adam([epsilon], lr=args.stepsize)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=args.stepsize_iters,
                                                gamma=args.stepsize_ratio)

    frozen_len = args.frozen_length

    y_logits_ = None
    noise_std = 0.0

    ## Encode x beforehand
    assert args.prefix_length <= 0, "The current code does not support prefix-length > 0"
    soft_forward_x = x_onehot[:, -1:, :]  # The last token of x is used in soft_forward
    if x_t.shape[1] == 1:
        x_model_past = None
    else:
        x_model_outputs = model(x_t[:, :-1])
        x_model_past = x_model_outputs.past_key_values
        # print(x_model_past[0])
        # x_model_past = [_.detach() for _ in x_model_past]

    # For right to left model rl_reverse_index = [length-1, length-2,.....,0]
    rl_reverse_index = torch.arange(y_logits.shape[1] - 1, -1, -1)

    mask_t = None

    for iter in range(args.num_iters):
        optim.zero_grad()
        y_logits_ = y_logits + epsilon

        soft_forward_y = y_logits_ / 0.001
        if args.straight_through:
            if mask_t is None:
                soft_forward_y = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_  #[batch_size, length, vocab_size]
            else:
                soft_forward_y = top_k_filter_3d(y_logits_, args.topk, mask=mask_t, extra_mask=z_mask) / 0.001

        # soft_forward_x: [batch_size, 1, vocab_size]   Last token of x
        # soft_forward_y: [batch_size, length, vocab_size]
        # x_past: Context to start
        y_logits_t = soft_forward(model, soft_forward_x, soft_forward_y, x_past=x_model_past)
        # y_logits_t: [batch_size,length, vocab_size]

        if args.topk == 0:
            mask_t = None
        else:
            _, indices_t = torch.topk(y_logits_t, args.topk)    #indices_t = [batch_size, length, topk]
            mask_t = torch.zeros_like(y_logits_t).scatter_(2, indices_t, 1) #mask_t = [batch_size, length, vocab_size]

        # Compute loss, gradients, and update.
        lr_nll_loss = soft_nll(
            top_k_filter_3d(y_logits_t / args.output_lgt_temp, args.topk, extra_mask=z_mask),
            y_logits_ / args.input_lgt_temp)
        

        # Reverse GPT2 Pass
        if args.lr_nll_portion == 1.0:
            rl_nll_loss = lr_nll_loss
        else:
            yz_logits_rev = torch.flip(torch.cat([y_logits_, z_onehot], dim=1), [1])
            yz_logits_rev_t = soft_backward(model_back, yz_logits_rev / 0.00001)
            yz_logits_rev_rev_t = torch.flip(yz_logits_rev_t, [1])
            yz_logits_rev_rev_t = yz_logits_rev_rev_t[:, :, 1:y_logits_.shape[-1] + 1]
            yz_logits_rev_rev_t_ = yz_logits_rev_rev_t[:, :y_logits_.shape[1], :]

            tmp_logits = yz_logits_rev_rev_t_
            repetition_mask = torch.cat([F.softmax(tmp_logits[:, 1:, :], dim=-1),
                                            torch.zeros_like(tmp_logits[:, -1:, :])], dim=1)
            yz_logits_rev_rev_t_ = yz_logits_rev_rev_t_ - repetition_mask * 1e4
            yz_logits_rev_rev_t_ = yz_logits_rev_rev_t_.detach()

            rl_nll_loss = soft_nll(
                top_k_filter_3d(yz_logits_rev_rev_t_ / args.rl_output_lgt_temp, args.rl_topk),
                y_logits_ / args.input_lgt_temp)
            

        if "lexical" in args.mode:
            soft_forward_y_ = (y_logits_.detach() / 0.3 - y_logits_).detach() + y_logits_
            xyz_logits, xy_length = soft_forward_xyz(model, soft_forward_x, soft_forward_y_, z_onehot)
            # xyz_logits: [batch_size, len_x+length+len_z, vocab_size]
            # xy Length : len_x+length

            # Reshaping
            bz = args.batch_size
            lg = xyz_logits.shape[1]
            st = xy_length - 1
            ed = xyz_logits.shape[1] - 1
            xyz_logits = xyz_logits.view(-1, xyz_logits.shape[-1])
            z_logits = torch.cat([xyz_logits[bi * lg + st:bi * lg + ed, :] for bi in range(bz)], dim=0)
            # [batch_size, len_z, vocab_size]
            c_loss_1 = torch.nn.CrossEntropyLoss(reduction='none')(
                z_logits,
                z_t.view(-1))
            c_loss_1 = c_loss_1.view(args.batch_size, -1).mean(-1)

            c_loss_2 = batch_log_bleulosscnn_ae(
                decoder_outputs=y_logits_.transpose(0, 1),
                target_idx=zz_t,
                ngram_list=[1],
                device= device
            )
            c_loss = c_loss_1 + args.abductive_c2_weight * c_loss_2

        loss = (1.0 - args.constraint_weight) * args.lr_nll_portion * lr_nll_loss \
               + (1.0 - args.constraint_weight) * (1 - args.lr_nll_portion) * rl_nll_loss \
               + args.constraint_weight * c_loss
        loss = loss.mean()

        if iter < args.num_iters - 1:  # so that the mask_t at the last iteration will not change
            loss.backward()
            optim.step()
            scheduler.step()  # turn off the scheduler
            last_lr = scheduler.get_last_lr()[0]

        # Top K sampling 
        if args.verbose and ((iter + 1) % args.print_every == 0 or iter == 0 or iter + 1 == args.num_iters):
            text, _, _ = decode_with_model_topk(
                model, y_logits_, args.topk, soft_forward_x, x_model_past, tokenizer, extra_mask=z_mask)
            for bi in range(args.batch_size):
                if "lexical" in args.mode:
                    print(
                        "%d, loss: %.4f, lr_nll_loss: %.4f, rl_nll_loss: %.4f,  c_loss_2: %.4f, lr: %.4f, |%s|" % (
                            iter + 1, loss.item(), lr_nll_loss[bi].item(), rl_nll_loss[bi].item(),
                            c_loss_2[bi].item(), last_lr, text[bi]))
                    # print("%d, loss: %.4f, lr_nll_loss: %.4f, rl_nll_loss: %.4f, c_loss_1: %.4f, c_loss_2: %.4f, lr: %.4f, |%s|" % (iter + 1, loss.item(), lr_nll_loss[bi].item(), rl_nll_loss[bi].item(), c_loss_1[bi].item(), c_loss_2[bi].item(), last_lr, text[bi]))

            print()

        # if args.wandb:
        #     wandb.log(
        #         {"Loss": loss.item(),
        #          "left-to-right nll loss": lr_nll_loss.item(),
        #          "right-to-left nll loss": rl_nll_loss.item(),
        #          "constraint loss": c_loss,
        #          "Gassian_Noise_STD": noise_std,
        #          "LR": last_lr,
        #          "Gradient": torch.norm(epsilon.grad).detach().clone().data.cpu().numpy()}
        #     )

        ## noise
        if iter < args.num_iters - 1:

            large_noise_iters = [int(_) for _ in args.large_noise_iters.split(',')]
            large_gs_stds = [float(_) for _ in args.large_gs_std.split(',')]
            noise_std = 0.
            if iter % args.noise_iters == 0:
                noise_last = True
                for ni in range(len(large_noise_iters)):
                    if iter < large_noise_iters[ni]:
                        noise_last = False
                        break
                if noise_last:
                    noise_std = args.gs_std
                else:
                    noise_std = large_gs_stds[ni]

                noise = torch.normal(mean=args.gs_mean, std=noise_std, size=epsilon.size(),
                                     device=device, requires_grad=False)
                if args.win_anneal_iters >= 0 and iter >= args.win_anneal_iters:
                    zeros = torch.zeros_like(noise)
                    noise_mix = torch.cat([zeros[:, :frozen_len], noise[:, frozen_len:]], dim=1)
                    y_logits = y_logits + noise_mix
                else:
                    y_logits = y_logits + noise

    # if args.wandb:
    #     wandb.finish()

    text, _, last_text_ids = decode_with_model_topk(
        model, y_logits_, args.topk, soft_forward_x, x_model_past, tokenizer, extra_mask=z_mask)

    last_rank_loss = model(input_ids=last_text_ids, labels=last_text_ids).loss
    last_rank_loss = last_rank_loss.detach().clone().data.cpu().numpy()
    text_post = post_process(last_text_ids, model, args.max_length, args.length, tokenizer, device)
    ppl_last = np.exp(last_rank_loss)

    if args.verbose:
        for bi in range(args.batch_size):
            print("[final]: %s\n%.4f" % (text[bi], ppl_last))
            print("[final complete sentence]: %s\n" % text_post[bi])

    return ppl_last, text, text_post


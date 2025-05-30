import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from transformers import RobertaTokenizer, Adafactor, AutoConfig,AutoTokenizer,BartTokenizer
import torch.optim as optim
import gc
import os
from functools import partial
import argparse
import random
import numpy as np
from compare_mt.rouge.rouge_scorer import RougeScorer
from sentence_transformers import SentenceTransformer, util


from config import *
from utils import Recorder
from data_utils import to_cuda, collate_mp, SumDataset
from model import MultiMarginLoss, MultiNllLoss, LexiSem
from torch.cuda.amp import autocast, GradScaler

import wandb
FULL_DATA_DIR="/home/aitech/ssd/projects/Ranker/Balsum_sub/SegmentedDataset"
def get_optimizer(model, lr, adam_eps=1e-8, weight_decay=0.0)->torch.optim.Optimizer:
    """
        Adafactor Optimizer
    """
    optimizer = Adafactor(
        model.parameters(),
        lr=lr,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,    
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    return optimizer



def evaluation(args):
    # setup 
    if args.config == "cnndm":
        cnndm_setting(args)
    elif args.config == "xsum":
        xsum_setting(args)
    elif args.config == "meqsum":
        meqsum_setting(args)
    elif args.config == "samsum":
        samsum_setting(args)
    elif args.config == "reddit":
        reddit_setting(args)

    tok = RobertaTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
    #test_set = SumDataset(f"/home/aitech/ssd/projects/Ranker/Balsum_sub/SegmentedDataset/meqsum/test", args.model_type, is_test=True, max_len=args.max_len, is_sorted=False, max_num=args.max_num, is_untok=True)
    test_set = SumDataset(f"{FULL_DATA_DIR}/{args.dataset}/test", args.model_type,args.tokenizer, is_test=True, max_len=args.max_len, is_sorted=False, max_num=args.max_num, is_untok=True)

    dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # Build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    model_hidden_size = AutoConfig.from_pretrained(model_path).hidden_size
    scorer = LexiSem(model_path, tok.pad_token_id, tok.cls_token_id, model_hidden_size, args.temp, args.gpuid[0])
    if args.cuda:
        scorer = scorer.cuda()
    scorer.load_state_dict(torch.load(os.path.join(f"./caches/cache_{args.dataset}/{args.model_pt}", "best_model.bin"), map_location=f'cuda:{args.gpuid[0]}'))
    scorer.eval()
    eval_dir = f"./results/result_{args.dataset}/{args.model_pt}"

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    print("eval_dir: ",eval_dir)
    mkdir(eval_dir)
    mkdir(f'{eval_dir}/reference')
    mkdir(f'{eval_dir}/candidate')

    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    cnt = 0

    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, args.gpuid[0])
            
            samples = batch['data']
            output = scorer(text_id=batch['src_input_ids'], candidate_ids=batch['candidate_ids'],abstract_ids=batch['abstract_ids'], is_test=True)
            
            similarity = output['score']
            similarity = similarity.cpu().numpy()
            max_ids = similarity.argmax(1)
            for j in range(similarity.shape[0]):       
                sample = samples[j]
                sents = sample['candidates'][max_ids[j]][0]
                score = rouge_scorer.score("\n".join(sample['abstract']), "\n".join(sents))

                rouge1 += score['rouge1'].fmeasure
                rouge2 += score['rouge2'].fmeasure
                rougeLsum += score['rougeLsum'].fmeasure

                with open(f'./{eval_dir}/candidate/{cnt}.dec', 'w', encoding='utf-8') as f:
                    for s in sents:
                        print(s, file=f)
                
                with open(f'./{eval_dir}/reference/{cnt}.ref', 'w', encoding='utf-8') as f:
                    for s in sample['abstract']:
                        print(s, file=f)

                cnt += 1

            # Trigger garbage collection
            del batch, samples, output, similarity
            gc.collect()
            torch.cuda.empty_cache()

    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    print("rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))


def test(dataloader, scorer, args, gpuid):
    scorer.eval()
    if args.cuda:
        device = f'cuda:{gpuid}'
    else:
        device = 'cpu'


    val_loss = 0
    cnt = 0
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    
    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)

            samples = batch['data']
              
            output = scorer(text_id=batch['src_input_ids'],candidate_ids=batch['candidate_ids'],abstract_ids=batch['abstract_ids'],is_test=True)

            similarity = output['score']
            similarity = similarity.cpu().numpy()

            if i % 1000 == 0:
                print(f'Validation similarity : {similarity[0]}')
            
            max_ids = similarity.argmax(1)
            for j in range(similarity.shape[0]):    
                sample = samples[j]
                sents = sample['candidates'][max_ids[j]][0]
                score = rouge_scorer.score("\n".join(sample['abstract']), "\n".join(sents))
                rouge1 += score['rouge1'].fmeasure
                rouge2 += score['rouge2'].fmeasure
                rougeLsum += score['rougeLsum'].fmeasure

                cnt += 1
            
            del similarity, output, batch
            torch.cuda.empty_cache()
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt

    scorer.train()

    if len(args.gpuid) > 1:
        rouge1 = torch.FloatTensor([rouge1]).to(device)
        dist.all_reduce(rouge1, op=dist.ReduceOp.SUM)
        rouge1 = rouge1.item() / len(args.gpuid)

        rouge2 = torch.FloatTensor([rouge2]).to(device)
        dist.all_reduce(rouge2, op=dist.ReduceOp.SUM)
        rouge2 = rouge2.item() / len(args.gpuid)

        rougeLsum = torch.FloatTensor([rougeLsum]).to(device)
        dist.all_reduce(rougeLsum, op=dist.ReduceOp.SUM)
        rougeLsum = rougeLsum.item() / len(args.gpuid)

    # for only debug
    wandb.log({
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougeLsum': rougeLsum
    })

    return {
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougeLsum': rougeLsum
    }


def run(rank, args):
    # setup hyperparams
    if args.config == "cnndm":
        cnndm_setting(args)
    elif args.config == "xsum":
        xsum_setting(args)
    elif args.config == "meqsum":
        meqsum_setting(args)
    elif args.config == "samsum":
        samsum_setting(args)
    elif args.config == "reddit":
        reddit_setting(args)



    # init 
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # setup GPU
    gpuid = args.gpuid[0]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    if is_master:
        #print("created")
        id = len(os.listdir(f"./caches/cache_{args.config}"))
        recorder = Recorder(id, args.config)
        wandb.init(project=f'{args.config}')
        wandb.run.name = args.wandb
        wandb.run.save()

    # build dataloader
    #tok = RobertaTokenizer.from_pretrained(args.model_type)
    tok = BartTokenizer.from_pretrained(f"{args.tokenizer}")

    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=False)
    collate_fn_val = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)

    # Instance Weighting
    train_set = SumDataset(f"{FULL_DATA_DIR}/{args.dataset}/train", args.model_type,args.tokenizer, max_len=args.max_len, max_num=args.max_num, total_len=args.total_len, thre=args.thre, neg_size=args.neg_size)
    val_set = SumDataset(f"{FULL_DATA_DIR}/{args.dataset}/val", args.model_type,args.tokenizer, max_len=args.max_len, is_test=True, is_sorted=False, max_num=args.max_num, total_len=args.total_len)
    print('train_set : ',len(train_set))
    print('val_set : ',len(val_set))
    
    if is_mp:
        print("parallel")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=True, collate_fn=collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
    else:
        print("none parallel")
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
        print('dataloader',len(dataloader))
        print("valdataloader",len(val_dataloader))

    # build models 
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    model_hidden_size = AutoConfig.from_pretrained(model_path).hidden_size
    scorer = LexiSem(model_path, tok.pad_token_id, tok.cls_token_id, model_hidden_size, args.temp, gpuid)
    if len(args.model_pt) > 0:
        print("before loading the model")
        scorer.load_state_dict(torch.load(os.path.join(f"./caches/cache_{args.dataset}/{args.model_pt}", "best_model.bin"), map_location=f'cuda:{gpuid}'))
        print("After loading the model")
    if args.cuda:
        if is_mp:
            # Using DDP
            dist.init_process_group("gloo", rank=rank, world_size=world_size)
            scorer = nn.parallel.DistributedDataParallel(scorer.to(gpuid), [gpuid], find_unused_parameters=True)
        else:
            scorer = scorer.cuda()
    scorer.train()

    # debug
    if is_master:
        wandb.config.update(args)
        wandb.watch(scorer)

    ## Optimizer
    init_lr = args.max_lr/args.warmup_steps
    # 2) Adafactor
    s_optimizer = get_optimizer(model=scorer, lr=init_lr)

    # # scheduler
    updates_per_epoch = len(dataloader) // args.accumulate_step
    total_updates = updates_per_epoch * args.epoch

    # debug
    if is_master:
        recorder.write_config(args, [scorer], __file__)
        recorder.print(f'***** Optimizer & Learning rate *****')
        recorder.print(f'updates_per_epoch : {updates_per_epoch}, total_updates : {total_updates}')
        recorder.print(f'warmup_steps : {args.warmup_steps}')
        recorder.print(f'optimizer : {s_optimizer}')
        recorder.print()

    minimum_loss = 100
    all_step_cnt = 0
    if is_mp:
        if is_master:
            print("is master check")
            id = torch.FloatTensor([id]).to(gpuid)
        else:
            id = torch.zeros(1).to(gpuid)
        dist.all_reduce(id, op=dist.ReduceOp.SUM)
        id = int(id.item())

    # define evaluate function
    def eval_fn(rouge1, rouge2, rougeLsum):
        return 1 - ((rouge1 + rouge2 + rougeLsum)/3)
    
    scaler = GradScaler()
    # Open the file in write mode
    
          # Adjust the range as needed
            


    # start Training
    for epoch in range(args.epoch):
        with open(f"./caches/cache_{args.config}/Epochs.txt", "w") as file:
            file.write(f"Epoch:  {epoch}\n")
        s_optimizer.zero_grad()
        step_cnt = 0
        epoch_step = 0
        avg_loss = 0
        avg_ranking_loss = 0
        avg_nll_loss = 0
        print("current epoch :",epoch )

        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)
            step_cnt += 1


            # forward 
            with autocast():
                output = scorer(
                    text_id=batch['src_input_ids'],
                    candidate_ids=batch['candidate_ids'],
                    abstract_ids=batch['abstract_ids'],
                    neg_id=batch['negative_ids']
                )
                similarity, neg_similarity = output['score'], output['neg_score']

                ranking_loss = MultiMarginLoss(batch['costs'], similarity, args.margin)
                nll_loss = MultiNllLoss(similarity, neg_similarity, batch['positive_weights'],
                                        device=f'cuda:{gpuid}', is_IW=args.is_IW)

                loss = args.nll_scale * nll_loss + args.rank_scale * ranking_loss
                loss = loss / args.accumulate_step

            avg_loss += loss.item()
            avg_nll_loss += nll_loss.item() / args.accumulate_step
            avg_ranking_loss += ranking_loss.item() / args.accumulate_step
            scaler.scale(loss).backward()
            print(step_cnt)
            

            if step_cnt == args.accumulate_step:
                print("accumulate_step condition")
                # update
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(scorer.parameters(), args.grad_norm)

                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1

                # WARMUP adjust learning rate
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt*(args.warmup_steps**(-1.5)))
                for param_group in s_optimizer.param_groups:
                    param_group['lr'] = lr
                
                scaler.step(s_optimizer)
                scaler.update()
                s_optimizer.zero_grad()

                
            if epoch_step % args.report_freq == 0 and step_cnt == 0 and is_master:
                print("report_freq condition")
                # report stats
                print(f"id : {id}")
                print(f"similarity ({similarity.shape}) : {similarity[0]}")
                recorder.print("epoch : %d, batch : %d, avg loss : %.6f"%(epoch+1, epoch_step, avg_loss / args.report_freq))
                recorder.print("avg_ranking_loss : %.6f, avg_nll_loss : %.6f"%(avg_ranking_loss/args.report_freq, avg_nll_loss/args.report_freq))
                lr = s_optimizer.param_groups[0]['lr']
                recorder.print(f"learning rate : {lr:.6f}")
                recorder.print()

                # for only debug
                wandb.log({
                    'epoch': (epoch+1),
                    'batch': epoch_step,
                    'loss': (avg_loss / args.report_freq),
                    'learning_rate': lr,
                    'avg_ranking_loss': (avg_ranking_loss / args.report_freq),
                    'avg_nll_loss': (avg_nll_loss / args.report_freq)
                })

                avg_loss = 0
                avg_ranking_loss, avg_nll_loss = 0, 0
            del similarity, loss, output, batch
            torch.cuda.empty_cache()
            gc.collect()

            if all_step_cnt % args.eval_interval == 0 and all_step_cnt != 0 and step_cnt == 0:
                print("eval_interval condition")
                # evaluate model 
                result = test(val_dataloader, scorer, args, gpuid)
                loss = eval_fn(result['rouge1'], result['rouge2'], result['rougeLsum'])
                if loss < minimum_loss and is_master:
                    print("is master1")
                    minimum_loss = loss
                    if is_mp:
                        recorder.save(scorer.module, "best_model.bin")
                    else:
                        recorder.save(scorer, "best_model.bin")
                    recorder.print('best - epoch : %d, batch : %d'%(epoch, i/args.accumulate_step))

                if is_master:
                    print("is master1")
                    recorder.print("val ranking loss: %.6f"%(loss))
                    recorder.print("val ranking rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
                    %(result["rouge1"], result["rouge2"], result["rougeLsum"]))
                
                # save current model
                if is_master:
                    if is_mp:
                        recorder.save(scorer.module, "model_cur.bin")
                    else:
                        recorder.save(scorer, "model_cur.bin")
                torch.cuda.empty_cache()    
                gc.collect()



def main(args):
    if len(args.gpuid) > 1:
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args), nprocs=len(args.gpuid), join=True)
    else:
        run(0, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Parameter")
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--gpuid', nargs='+', type=int, default=[0], help='gpu ids')
    parser.add_argument('--log', '-l', action='store_true', help='logging')
    parser.add_argument('--config', type=str, help='config path')
    parser.add_argument('--model_pt', type=str, default=r"", help="model path")
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate model')
    parser.add_argument('-p', '--port', type=int, default=29500, help='port')
    parser.add_argument('--wandb', type=str, help='Project Name')
    args = parser.parse_args()
    
    if args.cuda is False:
        print('GPU needed !!')
    else:
        if args.evaluate:
            with torch.cuda.device(args.gpuid[0]):
                evaluation(args)
        elif len(args.gpuid) == 1:
            main(args)
        else:
            main(args)

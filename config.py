def xsum_setting(args):
    # default setting for xsum
    args.dataset = getattr(args, 'dataset', 'xsum')
    args.batch_size = getattr(args, 'batch_size', 4)
    args.epoch = getattr(args, 'epoch', 5)      
    args.report_freq = getattr(args, 'report_freq', 100)
    args.eval_interval = getattr(args, 'eval_interval', 1000)   
    args.accumulate_step = getattr(args, 'accumulate_step', 12)
    args.model_type = getattr(args, 'model_type', 'princeton-nlp/sup-simcse-roberta-large') # princeton-nlp/sup-simcse-roberta-large
    args.max_lr = getattr(args, 'max_lr', 2e-3)   
    args.seed = getattr(args, 'seed', 970903)
    args.datatype = getattr(args, 'datatype', 'with_neg_random')    # diverse
    args.max_len = getattr(args, 'max_len', 80)     # max length of summary
    args.max_num = getattr(args, 'max_num', 10)     # max number of candidate summaries
    args.total_len = getattr(args, 'total_len', 512)    # total length of source article
    args.gen_max_len = getattr(args, 'gen_max_len', 62)     # max length of generated summaries
    args.gen_min_len = getattr(args, 'gen_min_len', 11)
    args.grad_norm = getattr(args, 'grad_norm', 0)
    args.pretrained = getattr(args, 'pretrained', None)
    args.warmup_steps = getattr(args, "warmup_steps", 10000)
    args.margin = getattr(args, 'margin', 0.1)     
    args.temp = getattr(args, 'temp', 0.05)
    args.rank_scale = getattr(args, 'rank_scale', 10)       
    args.nll_scale = getattr(args, 'nll_scale', 0.1)         
    args.neg_size = getattr(args, 'neg_size', 4)
    args.thre = getattr(args, 'thre', 0)
    args.is_IW = getattr(args, 'is_IW', True)



def cnndm_setting(args):
    # default setting for cnndm
    args.dataset = getattr(args, 'dataset', 'cnndm')
    args.batch_size = getattr(args, 'batch_size',1)   
    args.epoch = getattr(args, 'epoch', 5)
    args.report_freq = getattr(args, 'report_freq', 2)
    args.eval_interval = getattr(args, 'eval_interval', 10)
    #If batch_size = 4 and accumulate_step = 12, then the model behaves as if it had a batch size of 4 × 12 = 48 before updating.
    args.accumulate_step = getattr(args, 'accumulate_step', 2)
    args.model_type = getattr(args, 'model_type','princeton-nlp/sup-simcse-roberta-large') # princeton-nlp/sup-simcse-roberta-large
    args.max_lr = getattr(args, 'max_lr', 2e-3)     
    args.seed = getattr(args, 'seed', 970903)
    args.datatype = getattr(args, 'datatype', 'with_neg_random')    # diverse
    args.max_len = getattr(args, 'max_len', 120)     # max length of summary
    args.max_num = getattr(args, 'max_num', 16)     # max number of candidate summaries
    args.total_len = getattr(args, 'total_len', 512)  # Roberta both max position is 512 
    args.gen_max_len = getattr(args, 'gen_max_len', 140)     # max length of generated summaries
    args.gen_min_len = getattr(args, 'gen_min_len', 55)
    args.grad_norm = getattr(args, 'grad_norm', 0)
    args.pretrained = getattr(args, 'pretrained', None)
    args.warmup_steps = getattr(args, "warmup_steps", 10000)
    args.margin = getattr(args, 'margin', 1)      
    args.temp = getattr(args, 'temp', 0.05)
    args.rank_scale = getattr(args, 'rank_scale', 10)      
    args.nll_scale = getattr(args, 'nll_scale', 0.1)     
    args.neg_size = getattr(args, 'neg_size', 4)
    args.thre = getattr(args, 'thre', 0)
    args.is_IW = getattr(args, 'is_IW', True)



def meqsum_setting(args):
    # default setting for cnndm
    args.dataset = getattr(args, 'dataset', 'meqsum')
    args.batch_size = getattr(args, 'batch_size',2)   
    args.epoch = getattr(args, 'epoch', 1000)
    args.report_freq = getattr(args, 'report_freq', 100)
    args.eval_interval = getattr(args, 'eval_interval', 100)
    #If batch_size = 4 and accumulate_step = 12, then the model behaves as if it had a batch size of 4 × 12 = 48 before updating.
    args.accumulate_step = getattr(args, 'accumulate_step', 12)
    args.model_type = getattr(args, 'model_type','princeton-nlp/sup-simcse-roberta-large') # princeton-nlp/sup-simcse-roberta-large
    args.tokenizer = getattr(args, 'tokenizer','princeton-nlp/sup-simcse-roberta-large')
    args.max_lr = getattr(args, 'max_lr', 4e-3)     
    args.seed = getattr(args, 'seed', 42)
    args.datatype = getattr(args, 'datatype', 'with_neg_random')    # diverse
    args.max_len = getattr(args, 'max_len', 120)     # max length of summary
    args.max_num = getattr(args, 'max_num', 10)     # max number of candidate summaries
    args.total_len = getattr(args, 'total_len', 512)  # Roberta both max position is 512 
    args.gen_max_len = getattr(args, 'gen_max_len', 70)     # max length of generated summaries
    args.gen_min_len = getattr(args, 'gen_min_len', 55)
    args.grad_norm = getattr(args, 'grad_norm', 0)
    args.pretrained = getattr(args, 'pretrained', None)
    args.warmup_steps = getattr(args, "warmup_steps", 6400)
    args.margin = getattr(args, 'margin', 1)      
    args.temp = getattr(args, 'temp', 0.05)
    args.rank_scale = getattr(args, 'rank_scale', 10)      
    args.nll_scale = getattr(args, 'nll_scale', 0.1)     
    args.neg_size = getattr(args, 'neg_size', 10)
    args.thre = getattr(args, 'thre', 0)
    args.is_IW = getattr(args, 'is_IW', True)




def samsum_setting(args):
    # default setting for cnndm
    args.dataset = getattr(args, 'dataset', 'samsum')
    args.batch_size = getattr(args, 'batch_size',2)   
    args.epoch = getattr(args, 'epoch', 100)
    args.report_freq = getattr(args, 'report_freq', 100)
    args.eval_interval = getattr(args, 'eval_interval', 50)
    #If batch_size = 4 and accumulate_step = 12, then the model behaves as if it had a batch size of 4 × 12 = 48 before updating.
    args.accumulate_step = getattr(args, 'accumulate_step', 12)
    args.model_type = getattr(args, 'model_type','shogun-the-great/finetuned-bart-samsum') # princeton-nlp/sup-simcse-roberta-large
    args.tokenizer = getattr(args, 'tokenizer','shogun-the-great/finetuned-bart-samsum')
    args.max_lr = getattr(args, 'max_lr', 2e-3)     
    args.seed = getattr(args, 'seed', 970903)
    args.datatype = getattr(args, 'datatype', 'with_neg_random')    # diverse
    args.max_len = getattr(args, 'max_len', 80)     # max length of summary
    args.max_num = getattr(args, 'max_num', 16)     # max number of candidate summaries
    args.total_len = getattr(args, 'total_len', 512)  # Roberta both max position is 512 
    args.gen_max_len = getattr(args, 'gen_max_len', 82)     # max length of generated summaries
    args.gen_min_len = getattr(args, 'gen_min_len', 55)
    args.grad_norm = getattr(args, 'grad_norm', 0)
    args.pretrained = getattr(args, 'pretrained', None)
    args.warmup_steps = getattr(args, "warmup_steps", 6400)
    args.margin = getattr(args, 'margin', 1)      
    args.temp = getattr(args, 'temp', 0.05)
    args.rank_scale = getattr(args, 'rank_scale', 10)      
    args.nll_scale = getattr(args, 'nll_scale', 0.1)     
    args.neg_size = getattr(args, 'neg_size', 10)
    args.thre = getattr(args, 'thre', 0)
    args.is_IW = getattr(args, 'is_IW', True)





def reddit_setting(args):
    # default setting for cnndm
    args.dataset = getattr(args, 'dataset', 'reddit')
    args.batch_size = getattr(args, 'batch_size',4)   
    args.epoch = getattr(args, 'epoch', 5)
    args.report_freq = getattr(args, 'report_freq', 100)
    args.eval_interval = getattr(args, 'eval_interval', 1000)
    #If batch_size = 4 and accumulate_step = 12, then the model behaves as if it had a batch size of 4 × 12 = 48 before updating.
    args.accumulate_step = getattr(args, 'accumulate_step', 12)
    args.model_type = getattr(args, 'model_type','princeton-nlp/sup-simcse-roberta-large') # princeton-nlp/sup-simcse-roberta-large
    args.max_lr = getattr(args, 'max_lr', 2e-3)     
    args.seed = getattr(args, 'seed', 970903)
    args.datatype = getattr(args, 'datatype', 'with_neg_random')    # diverse
    args.max_len = getattr(args, 'max_len', 120)     # max length of summary
    args.max_num = getattr(args, 'max_num', 16)     # max number of candidate summaries
    args.total_len = getattr(args, 'total_len', 512)  # Roberta both max position is 512 
    args.gen_max_len = getattr(args, 'gen_max_len', 140)     # max length of generated summaries
    args.gen_min_len = getattr(args, 'gen_min_len', 55)
    args.grad_norm = getattr(args, 'grad_norm', 0)
    args.pretrained = getattr(args, 'pretrained', None)
    args.warmup_steps = getattr(args, "warmup_steps", 10000)
    args.margin = getattr(args, 'margin', 1)      
    args.temp = getattr(args, 'temp', 0.05)
    args.rank_scale = getattr(args, 'rank_scale', 10)      
    args.nll_scale = getattr(args, 'nll_scale', 0.1)     
    args.neg_size = getattr(args, 'neg_size', 4)
    args.thre = getattr(args, 'thre', 0)
    args.is_IW = getattr(args, 'is_IW', True)





import argparse
import os

class BaseOptions:
    """
    This class defines options used during training and testing    
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and testing."""
        parser.add_argument('--is_distributed',       action='store_true')
        parser.add_argument('--logdir',               type=str,     default='/home/likegiver/Desktop/codes/2023_4/clip_section/logs')
        parser.add_argument('--log_name',             type=str,     default='coop_log')
        parser.add_argument('--n_ctx',          type=int,     default=16)
        parser.add_argument('--class_token_position',             type=str,     default='end')
        
        parser.add_argument('--modeldir',            type=str,     default='/home/likegiver/Desktop/codes/2023_4/clip_section/saved_models')

        #### train ####
        parser.add_argument('--now_best',             type=float,   default=100)
        parser.add_argument('--load_pretrained',      action='store_true')
        parser.add_argument('--load_model_path',      type=str,     default='../train_results/checkpoints/train_latest.pth')
        parser.add_argument('--start_epoch',          type=int,     default=0)
        parser.add_argument('--batchsize',            type=int,     default=8,            help="batchsize for training and testing")
        parser.add_argument('--workers',              type=int,     default=4,              help="thread number for read images") 
        parser.add_argument('--epochs',               type=int,     default=1000,           help="epochs for train process")
        parser.add_argument('--loadOnInit',           action='store_true',                  help='set true if memory is very large')
        parser.add_argument('--pin_memo',             action='store_true',                  help='decide if pin_memory is True')
        parser.add_argument('--learning_rate',        type=float,   default=1e-3,           help="learning rate for training")
        parser.add_argument('--clip_learning_rate',   type=float,   default=1e-5,           help="learning rate for training")
        parser.add_argument('--print_freq',           type=int,     default=10)
        parser.add_argument('--eval_freq',            type=int,     default=10)
                #### optim #####
        parser.add_argument('--optim',                type=str,     default='adam',         help="adam, sgd, adamw")
        parser.add_argument('--lr_sheduler',          type=str,     default='cosine')
        parser.add_argument('--lr_sheduler_per_epoch',type=int,     default=50)
        parser.add_argument('--lr_warmup_step',       type=int,     default=5)

        parser.add_argument('--frozen_clip',          action='store_true')
        
        ####backbone clip networks####
        parser.add_argument('--backbone_path',        type=str,     default='/home/yushui/mc2/wuxun/wuxun/temporal_sentence_grounding/zero-shot-TSG/clip_based_zero_shot_TSG/clip_code/checkpoints/ViT-B-32.pt')
        parser.add_argument('--input_size',           type=int,     default='224',                                                   help="size of input image")

        ####visual prompt####
        parser.add_argument('--tsm',                  action='store_true')
        parser.add_argument('--visual_drop_out',      type=float,   default=0.0)
        parser.add_argument('--visual_emb_dropout',   type=float,   default=0.0)
        parser.add_argument('--joint',                action='store_true')
        parser.add_argument('--sim_header',           type=str,     default='Transf',                                                help="Transf   meanP  LSTM Conv_1D Transf_cls")

        ####PromptLearner####
        parser.add_argument('--CTX_INIT',             type=str,     default='a photo of a',                                          help="use given words to initialize context vectors")
        parser.add_argument('--N_CTX',                type=int,     default=5,                                                      help="number of context words (tokens) in prompts") 
        parser.add_argument('--VERB_TOKEN_POSITION',  type=str,     default='middle',                                                help="'middle' or 'end' or 'front'")
        parser.add_argument('--n_verb',               type=int,     default=5,                                                      help="number of verbs words (tokens) in prompts") 

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            print("parser initialized!")

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt): 
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        return message

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        return self.gather_options()
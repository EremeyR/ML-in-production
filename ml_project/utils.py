import argparse

from config import lr

def args_parser():
    parser = argparse.ArgumentParser(description='Listen Attend and Spell')
    # general
    parser.add_argument('--input-dim', type=int, default=40, help='input dimension')
    parser.add_argument('--encoder-hidden-size', type=int, default=512, help='encoder hidden size')
    parser.add_argument('--decoder-hidden-size', type=int, default=1024, help='decoder hidden size')
    parser.add_argument('--num-layers', type=int, default=4, help='number of encoder layers')
    parser.add_argument('--embedding-dim', type=int, default=512, help='embedding dimension')
    parser.add_argument('--end-epoch', type=int, default=150, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=lr, help='start learning rate')
    parser.add_argument('--lr-step', type=int, default=5, help='period of learning rate decay')
    parser.add_argument('--optimizer', default='adam', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size in each context')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    args = parser.parse_args()
    return args
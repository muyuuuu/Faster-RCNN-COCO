import torch


def writelog(file, log_info):
    '''
    write log informantion to file
    '''
    with open(file, 'a') as f:
        f.write(log_info + '\n')

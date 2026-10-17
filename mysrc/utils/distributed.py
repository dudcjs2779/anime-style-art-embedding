
def is_global_master(args):
    return args.rank == 0

def is_local_master(args):
    return args.local_rank == 0

def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)
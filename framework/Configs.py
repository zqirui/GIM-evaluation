from dataclasses import dataclass

@dataclass
class EvalConfig():
    """
    Evaluation Configuration Class
    """
    # metrics
    inception_score: bool = False
    fid: bool = False
    kid: bool = False
    is_infinity: bool = False
    fid_infinity: bool = False
    clean_fid: bool = False
    clean_kid: bool = False
    prd: bool = False
    prc: bool = False
    ls: bool = False
    c2st_1nn: bool = False
    # metric parameter
    # IS
    is_splits: int = 10
    # FID

    # KID
    kid_subsets: int = 100
    kid_subset_size: int = 1000
    kid_degree: int = 3
    kid_coef0: int = 1
    # PRC
    prc_neighborhood: int = 3
    prc_batch_size: int = 10000




@dataclass
class PlatformConfig():
    """
    General Platform Configuration Class
    """ 
    # Output verbose
    verbose: bool = True
    # torch-fidelity RAM saving option
    save_cpu_ram: bool = True
    # overall batch size 
    batch_size: int = 64
    # cuda suppoert
    cuda: bool = True
    # compare also real to real
    compare_real_to_real: bool = True

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
    # KID
    kid_subsets: int = 100
    kid_subset_size: int = 1000
    kid_degree: int = 3
    kid_coef0: int = 1
    # PRC
    prc_neighborhood: int = 3
    prc_batch_size: int = 10000
    # FID infty
    fid_infty_batch_size: int = 128
    # IS infty
    is_infty_batch_size: int = 128
    # clean FID
    clean_fid_batch_size: int = 32
    # clean KID
    clean_kid_batch_size: int = 32
    # LS
    ls_n_samples: int = 0 # num of samples for computation, if this is 0 take k-fold approach
    ls_n_folds: int = 5 # folds for cross validation
    ls_plot_distances: bool = False # plot histogram of distances

    def __post_init__(self):
        if self.ls:
            assert self.ls_n_folds != 0 or self.ls_n_samples != 0, "N folds and n samples unspecified! Either needs to be > 0"
            assert (self.ls_n_folds == 0 and self.ls_n_samples != 0) or (self.ls_n_folds != 0 and self.ls_n_samples == 0), "Both options n fold cross validation and n samples specified! Only one is supported! Please set one Option to 0"
    




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
    # cpu threads to use 
    num_worker : int = 0
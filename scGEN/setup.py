from opt import args


def setup_args(dataset_name="cora"):
    args.dataset = dataset_name
    args.device = "cuda:0"
    args.acc = args.nmi = args.ari = args.f1 = 0


    args.n_input = 500
    args.dims = 1500
    args.activate = 'ident'

    print("---------------------")
    print("runs: {}".format(args.runs))
    print("dataset: {}".format(args.dataset))
    print("Hard-sample focusing factor: {}".format(args.gamma))
    print("learning rate: {}".format(args.lr))
    print("---------------------")

    return args

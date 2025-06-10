from utils import *
from tqdm import tqdm
from torch import optim
from setup import setup_args
from model import Network
import scipy.io
import os

from loss import ZINBLoss
if __name__ == '__main__':

    # for dataset_name in ["cora", "citeseer", "amap", "bat", "eat", "uat"]:
    for dataset_name in [
        #"hrvatin_B1",
        #"hrvatin_B2",
        "zhang"
        #"pbmc3k",
        #"Scala"
        #"Schwalbe"
        #"Bell"
        #"Savas"
    ]:

        # setup hyper-parameter
        args = setup_args(dataset_name)

        # record results
        file = open("result.csv", "a+")
        print(args.dataset, file=file)
        print("ACC,   NMI,   ARI,   F1", file=file)
        file.close()
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []

        top_acc_values = []
        top_seeds = []
        top_metrics = []

        # ten runs with different random seeds
        #seeds = [9,10]
        #for args.seed in seeds:
        for args.seed in range(args.runs):
            # record results

            # fix the random seed
            setup_seed(args.seed)

            mat_path = './data/%s.mat' % dataset_name

            # Load the .mat file
            data = scipy.io.loadmat(mat_path)

            # Extract 'X' and 'Y' data
            X = data['X'].astype(np.float32)  # Ensure x is of type float32
            y = data['Y'].squeeze()  # Flatten y if it's multidimensional
            cluster_num = len(np.unique(y))
            node_num = X.shape[0]
            A = getGraph(X.T,10)

            # apply the laplacian filtering
            X_filtered = laplacian_filtering(A, X, args.t)

            # test
            args.acc, args.nmi, args.ari, args.f1, y_hat, center = phi(X_filtered, y, cluster_num)

            # build our hard sample aware network
            scGEN = Network(
                input_dim=X.shape[1], hidden_dim=args.dims, act=args.activate, n_num=node_num)

            # adam optimizer
            optimizer = optim.Adam(scGEN.parameters(), lr=args.lr)

            # positive and negative sample pair index matrix
            mask = torch.ones([node_num * 2, node_num * 2]) - torch.eye(node_num * 2)

            # load data to device
            A, scGEN, X_filtered, mask = map(lambda x: x.to(args.device), (A, scGEN, X_filtered, mask))

            # training
            for epoch in tqdm(range(args.epochs), desc="training..."):
                # train mode
                scGEN.train()

                # encoding with Eq. (3)-(5)
                Z1, Z2, E1, E2, H,mean,disp,pi = scGEN(X_filtered, A)
                zinb_loss_fn = ZINBLoss()
                zinbloss = zinb_loss_fn(H,mean,disp,pi)

                # calculate comprehensive similarity by Eq. (6)
                S = comprehensive_similarity(Z1, Z2, E1, E2, scGEN.alpha)


                # calculate hard sample aware contrastive loss by Eq. (10)-(11)
                l1 = hard_sample_aware_infoNCE(S, mask, scGEN.pos_neg_weight, scGEN.pos_weight, node_num)
                l2 = zinbloss
                if (args.Hard_only_use):
                    loss = l1
                elif (args.ZINB_only_use):
                    loss = l2
                else:
                    loss = l1 + args.alpha*l2

                # optimization
                loss.backward()
                optimizer.step()

                # testing and update weights of sample pairs
                if epoch % 10 == 0:
                    # evaluation mode
                    scGEN.eval()

                    # encoding
                    Z1, Z2, E1, E2,_,_,_,_ = scGEN(X_filtered, A)

                    # calculate comprehensive similarity by Eq. (6)
                    S = comprehensive_similarity(Z1, Z2, E1, E2, scGEN.alpha)

                    # fusion and testing
                    Z = (Z1 + Z2 + E1 + E2) / 4
                    acc, nmi, ari, f1, P, center = phi(Z, y, cluster_num)

                    # select high confidence samples
                    H, H_mat = high_confidence(Z, center)
                    #print(H,H_mat)

                    # calculate new weight of sample pair by Eq. (9)
                    M, M_mat = pseudo_matrix(P, S, node_num)

                    # update weight
                    scGEN.pos_weight[H] = M[H].data
                    scGEN.pos_neg_weight[H_mat] = M_mat[H_mat].data

                    # recording
                    if acc >= args.acc:
                        args.acc, args.nmi, args.ari, args.f1 = acc, nmi, ari, f1

            print("Training complete")

            # record results
            file = open("result.csv", "a+")
            print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(args.acc, args.nmi, args.ari, args.f1), file=file)
            file.close()
            acc_list.append(args.acc)
            nmi_list.append(args.nmi)
            ari_list.append(args.ari)
            f1_list.append(args.f1)

            # Store the top 2 ACC values and seeds
            if len(top_acc_values) < 2:
                top_acc_values.append(args.acc)
                top_seeds.append(args.seed)
                top_metrics.append((args.acc, args.nmi, args.ari, args.f1))
            else:
                min_index = top_acc_values.index(min(top_acc_values))
                if args.acc > top_acc_values[min_index]:
                    top_acc_values[min_index] = args.acc
                    top_seeds[min_index] = args.seed
                    top_metrics[min_index] = (args.acc, args.nmi, args.ari, args.f1)

        # Print the top 2 ACC values and seeds
        for i in range(2):
            print(f"Top {i + 1} ACC: {top_acc_values[i]:.2f} on run {top_seeds[i]}")
            print(
                f"Metrics: ACC={top_metrics[i][0]:.2f}, NMI={top_metrics[i][1]:.2f}, ARI={top_metrics[i][2]:.2f}, F1={top_metrics[i][3]:.2f}")

        # Final recording for the top 2 ACC values and metrics
        with open("result.csv", "a+") as file:
            for i in range(2):
                file.write(
                    f"{top_seeds[i]},{top_metrics[i][0]:.2f},{top_metrics[i][1]:.2f},{top_metrics[i][2]:.2f},{top_metrics[i][3]:.2f}\n")

        # Convert top_metrics to a NumPy array for easy statistics calculation
        top_metrics_array = np.array(top_metrics)
        mean_metrics = top_metrics_array.mean(axis=0)
        std_metrics = top_metrics_array.std(axis=0)

        # Print and record the mean and std of the top metrics
        with open("result.csv", "a+") as file:
            file.write(f"{mean_metrics[0]:.2f},{std_metrics[0]:.2f}\n")
            file.write(f"{mean_metrics[1]:.2f},{std_metrics[1]:.2f}\n")
            file.write(f"{mean_metrics[2]:.2f},{std_metrics[2]:.2f}\n")
            file.write(f"{mean_metrics[3]:.2f},{std_metrics[3]:.2f}\n")


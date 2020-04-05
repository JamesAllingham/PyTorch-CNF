"""Script to reproduce figure 5 of "Neural Ordinary Differential Equations" by Chen et al. (2018)"""
import os
import argparse

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles

import torch
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint

from cnf import CNF

parser = argparse.ArgumentParser(
    description='Script to reproduce figure 5 of \
                "Neural Ordinary Differential Equations" by Chen et al. (2018)'
)
parser.add_argument('--dataset', type=str, help='Which dataset to train on.',
                    choices=["moons", "circles"])
parser.add_argument('--GPU', type=str, help='Which GPU to use. (default: None)',
                    default=None)
parser.add_argument('--save_dir', type=str, help='Where to save the models. (default: "../saves")',
                    default="../saves")
parser.add_argument('--num_samples', type=int, help='Number of samples to draw per training iteration. (default: 512)',
                    default=512)
parser.add_argument('--hidden_dim', type=int, help='Size of the CNF hidden dim. (default: 32)',
                    default=32)
parser.add_argument('--width', type=int, help='Size of the CNF width. (default: 64)',
                    default=64)
parser.add_argument('--epochs', type=int, help='Number of training rounds. (default: 1000)',
                    default=1000)
parser.add_argument('--train', help='Train rather than load the best model.',
                    default=False, action="store_true")

args = parser.parse_args()

if args.GPU is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch(num_samples, dataset):
    if dataset == 'moons':
        data_func = make_moons
        kwargs = {"noise": 0.07}
        modifier = lambda x: x - [0.5, 0.25]
    elif dataset == 'circles':
        data_func = make_circles
        modifier = lambda x: x / 1.25 * 2
        kwargs = {"noise": 0.06, "factor": 0.5}
    else:
        raise Exception(f'Dataset {dataset} not implemented.')

    points, _ = data_func(n_samples=num_samples, **kwargs)
    points = modifier(points)
    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)

    return(x, logp_diff_t1)


def calc_loss(odefunc, x, logp_diff_t1, t0, t1, p_z0):
    z_t, logp_diff_t = odeint(
        odefunc,
        (x, logp_diff_t1),
        torch.tensor([t1, t0]).type(torch.float32).to(device),
        atol=1e-5,
        rtol=1e-5,
        method='dopri5',
    )

    z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

    logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
    loss = -logp_x.mean(0)

    return loss


def main():
    save_dir = args.save_dir + f"/{args.dataset}/{args.width}/{args.hidden_dim}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    t0 = 0
    t1 = 10

    p_z0 = torch.distributions.MultivariateNormal(loc=torch.tensor([0.0, 0.0]).to(device),
                                                  covariance_matrix=torch.tensor([[0.3, 0.0], [0.0, 0.3]]).to(device))

    odefunc = CNF(in_out_dim=2, hidden_dim=args.hidden_dim, width=args.width)
    if device != "cpu":
        odefunc = odefunc.cuda()

    optimizer = optim.Adam(odefunc.parameters(), lr=1e-3, weight_decay=0.)

    x_test, logp_diff_t1_test = get_batch(args.num_samples * 20, args.dataset)

    # Train model
    for itr in tqdm(range(args.epochs + 1)):
        optimizer.zero_grad()

        x, logp_diff_t1 = get_batch(args.num_samples, args.dataset)

        loss = calc_loss(odefunc, x, logp_diff_t1, t0, t1, p_z0)

        loss.backward()
        optimizer.step()

        best_loss = np.inf
        if itr % 100 == 0:
            z_t, logp_diff_t = odeint(
                odefunc,
                (x_test, logp_diff_t1_test),
                torch.tensor([t1, t0]).type(torch.float32).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )

            z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

            logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
            loss = -logp_x.mean(0)
            if loss < best_loss:
                best_loss = loss
                torch.save(odefunc.state_dict(),
                           f"{save_dir}/best_model.pt")

            plt.figure(figsize=(4, 4), dpi=200)
            plt.hist2d(*z_t0.detach().cpu().numpy().T,
                       bins=300, density=True, range=[[-2, 2], [-2, 2]])
            plt.axis('off')
            plt.gca().invert_yaxis()
            plt.margins(0, 0)

            plt.savefig(save_dir + f"/tgt_itr_{itr:05d}.jpg",
                        pad_inches=0, bbox_inches='tight')
            plt.close()

    odefunc.load_state_dict(torch.load(f"{save_dir}/best_model.pt"))

    # Generate evolution of sampled points
    z_t0 = p_z0.sample([30000]).to(device)
    logp_diff_t0 = torch.zeros(30000, 1).type(torch.float32).to(device)

    z_t, logp_diff_t = odeint(
        odefunc,
        (z_t0, logp_diff_t0),
        torch.tensor(np.linspace(t0, t1, 21)).to(device),
        atol=1e-5,
        rtol=1e-5,
        method='dopri5',
    )

    for (t, z) in zip(np.linspace(t0, t1, 21), z_t.detach().cpu().numpy()):
        plt.figure(figsize=(4, 4), dpi=200)
        plt.hist2d(*z.T,
                   bins=300, density=True, range=[[-2, 2], [-2, 2]])
        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.margins(0, 0)

        plt.savefig(save_dir + f"/samples_{t:f}.jpg",
                    pad_inches=0, bbox_inches='tight')
        plt.close()

    # Generate evolution of density
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

    z_t1 = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)

    z_t, logp_diff_t = odeint(
        odefunc,
        (z_t1, logp_diff_t1),
        torch.tensor(np.linspace(t1, t0, 21)).to(device),
        atol=1e-5,
        rtol=1e-5,
        method='dopri5',
    )

    for (t, z, logp_diff) in zip(np.linspace(t0, t1, 21), z_t, logp_diff_t):
        logp = p_z0.log_prob(z) - logp_diff.view(-1)

        plt.figure(figsize=(4, 4), dpi=200)
        plt.tricontourf(*z_t1.detach().cpu().numpy().T,
                        np.exp(logp.detach().cpu().numpy()), 200)
        plt.tight_layout()
        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.margins(0, 0)

        plt.savefig(save_dir + f"/density_{t:f}.jpg",
                    pad_inches=0, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()

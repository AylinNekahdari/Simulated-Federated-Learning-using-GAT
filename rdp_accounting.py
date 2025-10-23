import matplotlib.pyplot as plt
from opacus.accountants import RDPAccountant

noise_multiplier = 0.5
num_clients = 3
rounds = 6
delta = 1e-5

def compute_rdp_epsilon(noise_multiplier, num_clients, rounds, delta=1e-5):
    """
    Computes epsilon values across rounds using the RDP accountant.
    
    Args:
        noise_multiplier (float): The noise multiplier (σ).
        num_clients (int): Number of participating clients.
        rounds (int): Number of training rounds.
        delta (float): Privacy parameter δ.
    
    Returns:
        list: Epsilon values for each round.
    """
    accountant = RDPAccountant()
    epsilons = []

    for _ in range(rounds):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=1/num_clients)
        eps = accountant.get_epsilon(delta=delta)
        epsilons.append(eps)
    
    return epsilons


def plot_rdp_epsilon(epsilons, delta=1e-5):
    """
    Plots epsilon (ε) vs rounds.
    
    Args:
        epsilons (list): Epsilon values over rounds.
        delta (float): Privacy parameter δ.
    """
    rounds = list(range(1, len(epsilons) + 1))
    plt.figure(figsize=(7, 5))
    plt.plot(rounds, epsilons, marker='o')
    plt.title(f"Formal RDP Accounting (δ={delta})")
    plt.xlabel("Rounds")
    plt.ylabel("Epsilon (ε)")
    plt.grid(True)
    plt.show()

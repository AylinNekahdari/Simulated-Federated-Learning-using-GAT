import torch
import torch.nn.functional as F
import numpy as np
from opacus.accountants import RDPAccountant


def train_local(model, data, epochs=3, lr=0.005, device="cpu"):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, total_acc = 0, 0
    for _ in range(epochs):
        opt.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = criterion(out, data.y.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        preds = out.argmax(dim=1)
        total_loss += loss.item()
        total_acc += (preds == data.y.to(device)).float().mean().item()
    return model.state_dict(), total_loss/epochs, total_acc/epochs


def fedavg_state_dicts(state_dicts):
    avg = {}
    for k in state_dicts[0].keys():
        avg[k] = torch.mean(torch.stack([sd[k].float().cpu() for sd in state_dicts]), dim=0)
    return avg


def clip_vector(vec, max_norm):
    norm = vec.norm().item()
    return vec * (max_norm / (norm + 1e-9)) if norm > max_norm else vec


def state_dict_to_vector(state_dict):
    vecs = [v.detach().cpu().float().reshape(-1) for v in state_dict.values()]
    return torch.cat(vecs)


def vector_to_state_dict(template_state, vec):
    new_state = {}
    offset = 0
    for k, v in template_state.items():
        numel = v.numel()
        chunk = vec[offset: offset + numel].reshape(v.shape).to(v.device).type(v.dtype)
        new_state[k] = chunk
        offset += numel
    return new_state


def run_federated_training(global_model,
                           clients_data,
                           val_data,
                           rounds,
                           local_epochs,
                           lr,
                           use_dp,
                           clip_norm,
                           noise_multiplier,
                           device="cpu"):
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
    train_losses, val_losses, val_accs = [], [], []

    for r in range(1, rounds+1):
        local_states, local_losses, local_accs = [], [], []
        for cdata in clients_data:
            local_model = type(global_model)(global_model.gat1.in_channels).to(device)
            local_model.load_state_dict({k: v.clone().to(device) for k, v in global_state.items()})
            st, loss, acc = train_local(local_model, cdata, local_epochs, lr, device)
            local_states.append({k: v.cpu().clone() for k, v in st.items()})
            local_losses.append(loss)
            local_accs.append(acc)

            print(f"Round {r} | Client {i+1}: loss={loss:.4f}, acc={acc:.4f}")

        # DP aggregation
        global_vec = state_dict_to_vector(global_state)
        if use_dp:
            noisy_deltas = []
            for st in local_states:
                client_vec = state_dict_to_vector(st)
                delta = clip_vector(client_vec - global_vec, clip_norm)
                noise = torch.normal(0, noise_multiplier * clip_norm, delta.shape)
                noisy_deltas.append(delta + noise)
            avg_noisy = torch.mean(torch.stack(noisy_deltas), dim=0)
            new_vec = global_vec + avg_noisy
            new_state = vector_to_state_dict(global_state, new_vec)
        else:
            new_state = fedavg_state_dicts(local_states)

        global_state = {k: v.clone() for k, v in new_state.items()}
        global_model.load_state_dict(global_state)

        global_model.eval()
        with torch.no_grad():
            out_val = global_model(val_data.x.to(device), val_data.edge_index.to(device))
            val_loss = F.cross_entropy(out_val, val_data.y.to(device)).item()
            val_acc = (out_val.argmax(1) == val_data.y.to(device)).float().mean().item()

        train_losses.append(float(np.mean(local_losses)))
        val_losses.append(float(val_loss))
        val_accs.append(float(val_acc))

        print(f"✅ Round {r}: avg_train_loss={np.mean(local_losses):.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    return global_model, train_losses, val_losses, val_accs

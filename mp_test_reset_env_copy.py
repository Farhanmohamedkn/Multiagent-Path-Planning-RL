import torch.multiprocessing as mp
import torch
import sys

def foo(worker, tl):
    # Each worker prints the current sum of its assigned tensor
    print(f'Worker {worker}: Sum before addition = {tl[worker].sum().item()}')
    # Each worker adds its worker ID to all elements of its assigned tensor
    tl[worker] += worker + 1  # Adding (worker + 1) to each element
    print(f'Worker {worker}: Sum after addition = {tl[worker].sum().item()}')

def run_torch_procs(n_procs):
    mp.set_start_method('spawn', force=True)
    tl = [torch.zeros(5) for k in range(n_procs)]

    for t in tl:
        t.share_memory_()

    print("Before multiprocessing: Total Sum =")
    print(torch.cat(tl).sum().item())
    print(tl)

    procs = []
    for k in range(n_procs):
        p = mp.Process(target=foo, args=(k, tl))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    print("After multiprocessing: Total Sum =")
    print(torch.cat(tl).sum().item())

if __name__ == '__main__':
    n_procs = 2
    run_torch_procs(n_procs)

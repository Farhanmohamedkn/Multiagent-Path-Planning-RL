import torch.multiprocessing as mp
import torch

def foo(shared_tensor,a):
    print(f'Initial sum in {a}  worker:{shared_tensor.sum().item()}')
    shared_tensor += 1  # Each worker increments every element by 1
    print(f'Updated sum in {a}  worker: {shared_tensor.sum().item()}')

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    shared_tensor = torch.zeros(2)
    print(f'before mp:{shared_tensor}')
    shared_tensor.share_memory_()

    processes = []
    for a in range(4):
        p = mp.Process(target=foo, args=(shared_tensor,a))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f'Final sum in main process: {shared_tensor.sum().item()}')
    print(f'after mp:{shared_tensor}')

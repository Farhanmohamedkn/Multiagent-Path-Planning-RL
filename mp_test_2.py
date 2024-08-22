import torch.multiprocessing as mp
import torch

def foo(shared_dict, key):
    tensor = shared_dict[key]
    print(f'before sum in worker for: {key}: {tensor.sum().item()}')
    tensor += 1
    print(f'After sum in worker for: {key}: {tensor.sum().item()}')

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    shared_dict = manager.dict()
    print(f'dict {shared_dict}')

    # Create and store tensors in a managed dictionary
    shared_dict['tensor1'] = torch.zeros(1).share_memory_()
    shared_dict['tensor2'] = torch.ones(1).share_memory_()

    # print(f'before mp {shared_dict['tensor1']}')

    processes = []
    for key in shared_dict.keys():
        p = mp.Process(target=foo, args=(shared_dict, key))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f'Final sum for tensor1: {shared_dict["tensor1"].sum().item()}')
    print(f'Final sum for tensor2: {shared_dict["tensor2"].sum().item()}')
    # print(f'after mp {shared_dict['tensor1']}')

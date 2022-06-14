import numpy as np 



def get_list_item_number(torch_tensor):
    result = 1 
    shape_ = torch_tensor.detach().numpy().shape
    for i in shape_:
        result *= i 
    return result, shape_

import numpy as np 



def get_list_item_number(listA: list):
    result = 1 
    shape_ = listA.detach().numpy().shape
    for i in shape_:
        result *= i 
    return result, shape_

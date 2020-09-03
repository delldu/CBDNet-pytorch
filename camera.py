import os
# import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random

import pdb


def load_CRF():
    # CRF = scipy.io.loadmat('matdata/201_CRF_data.mat')
    # iCRF = scipy.io.loadmat('matdata/dorfCurvesInv.mat')
    # B_gl = CRF['B']
    # I_gl = CRF['I']

    # if os.path.exists('matdata/201_CRF_iCRF_function.mat')==0:
    #     CRF_para = np.array(CRF_function_transfer(I_gl, B_gl))
    #     iCRF_para = 1. / CRF_para
    #     scipy.io.savemat('matdata/201_CRF_iCRF_function.mat', {'CRF':CRF_para, 'iCRF':iCRF_para})
    # else:
    #     Bundle = scipy.io.loadmat('matdata/201_CRF_iCRF_function.mat')
    #     CRF_para = Bundle['CRF']
    #     iCRF_para = Bundle['iCRF']

    CRF = torch.load("/tmp/201_crf_dataset.pth")

    x = CRF['I'][100]
    y = CRF['B'][100]

    plt.plot(x,y)
    plt.show()

    # pdb.set_trace()


    return CRF_para, iCRF_para



# load_CRF()
torch.manual_seed(0) 

a = torch.randn(1, requires_grad=True)

CRF = torch.load("/tmp/201_crf_dataset.pth")
index = random.randint(0, 200)

x = CRF['I'][index]
y = CRF['B'][index]
lr = 0.001

for e in range(100):
    y_pred = torch.pow(x, a)

    loss = (y - y_pred).pow(2).sum()

    if(e % 10==0):
        print("Epoch: {} Loss: {}".format(e, loss.item()))


    loss.backward()
    with torch.no_grad():
        a -= lr * a.grad
        # Manually zero the gradients after updating weights
        a.grad.zero_()


print("model.a = ", a, "index: ", index)

y_pred = torch.pow(x, a).detach()

plt.plot(x,y)
plt.plot(x,y_pred)
plt.show()

# pdb.set_trace()


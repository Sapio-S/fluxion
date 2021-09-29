import numpy as np

for i in range(10):
    dic = np.load("/home/yuqingxie/autosys/code/fluxion/tmp_data_norm0910/"+str(i)+"_csv_scale.npy", allow_pickle=True).item()
    print(dic["frontend:0.90:MAX"], dic["frontend:0.90:MIN"])
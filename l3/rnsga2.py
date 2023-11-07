import numpy as np
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter

def dist_3d(a, b):
    return (b[0]-a[0])**2+(b[1]-a[1])**2+(b[2]-a[2])**2

def dist_2d1(a, b):
    return (b[0]-a[0])**2+(b[2]-a[2])**2

def dist_2d2(a, b):
    return (b[1]-a[1])**2+(b[2]-a[2])**2

def find_best_k(min_coord, all_f, k, deg):
    cnt = 0
    d = []
    size = all_f.shape[0]
    for i in range(size):
        if(deg==0):
            # print("Coord"+ str(all_f[i]) + str(min_coord))
            d.append([i, dist_3d(all_f[i], min_coord)])
            # print("Dist "+ str(dist_3d(all_f[i], min_coord)))
        elif(deg==1):
            d.append([i, dist_2d1(all_f[i], min_coord)])
        elif(deg==2):
            d.append([i, dist_2d2(all_f[i], min_coord)])
            # print("Dist "+ str(dist_2d2(all_f[i], min_coord)))
    d = np.array(d)
    d_sort = d[d[:, 1].argsort()]
    # print(d_sort)
    # print('List:')
    ans = []
    for i in range(k):
        # print(int(d_sort[i][0]))
        ans.append(int(d_sort[i][0]))
    ans = np.array(ans)
    return ans

# def find_best_k_erracc(min_coord, all_f, k, deg):
#     cnt = 0
#     d = []
#     size = all_f.shape[0]
#     for i in range(size):
#         d.append([i, all_f[i][2]-min_coord[2]])
#     d = np.array(d)
#     d_sort = d[d[:, 1].argsort()]

#     ans = []
#     for i in range(k):
#         ans.append(all_f[int(d_sort[i][0])])
#     ans = np.array(ans)
#     return ans
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr
    
all_f_l3 = np.load('all_f_l5.npy')
# all_f_l4 = np.load('all_f_l4.npy')
# all_f_l5 = np.load('all_f_l5.npy')

res_x_l3 = np.load("RNSGA2_l5_PDPvsAREA_X.npy")
res_f_l3 = np.load("RNSGA2_l5_PDPvsAREA_F.npy")

print(res_x_l3)
# res_x_l4 = np.load("RNSGA2_l4_PDPvsAREA_X.npy")
# res_f_l4 = np.load("RNSGA2_l4_PDPvsAREA_F.npy")

# res_x_l5 = np.load("RNSGA2_l5_PDPvsAREA_X.npy")
# res_f_l5 = np.load("RNSGA2_l5_PDPvsAREA_F.npy")

pdp_all_l3 = all_f_l3[:, 0]
area_all_l3 = all_f_l3[:, 1]
erracc_all_l3 = all_f_l3[:, 2]

res_pdp_l3 = res_f_l3[:, 0]
res_area_l3 = res_f_l3[:, 1]
res_erracc_l3 = res_f_l3[:, 2]

n_res_pdp_l3 = normalize(res_pdp_l3, 0, 1)
n_res_area_l3 = normalize(res_area_l3, 0, 1)
n_res_erracc_l3 = normalize(res_erracc_l3, 0, 1)
# print(res_pdp_l3)
# print(res_area_l3)
# print(res_erracc_l3)
# print("After")
# print(n_res_pdp_l3)
# print(n_res_area_l3)
# print(n_res_erracc_l3)

min_l3 = [min(n_res_pdp_l3), min(n_res_area_l3), min(n_res_erracc_l3)]
n_res_f_l3 = np.zeros(res_f_l3.shape)
n_res_f_l3[:, 0] = n_res_pdp_l3
n_res_f_l3[:, 1] = n_res_area_l3
n_res_f_l3[:, 2] = n_res_erracc_l3

# pdp_all_l4 = all_f_l4[:, 0]
# area_all_l4 = all_f_l4[:, 1]
# erracc_all_l4 = all_f_l4[:, 2]
# min_l4 = [min(pdp_all_l4), min(area_all_l4), min(erracc_all_l4)]

# pdp_all_l5 = all_f_l5[:, 0]
# area_all_l5 = all_f_l5[:, 1]
# erracc_all_l5 = all_f_l5[:, 2]
# min_l5 = [min(pdp_all_l5), min(area_all_l5), min(erracc_all_l5)]

k = 3
idx_all = find_best_k(min_l3, n_res_f_l3, k, 0)
idx_pdpvserr = find_best_k(min_l3, n_res_f_l3, k, 1)
idx_arvserr = find_best_k(min_l3, n_res_f_l3, k, 2)

best_pnts_all = []
best_pnts_pdpvserr = []
best_pnts_arvserr = []
best_all_mult = []
best_pdpvserr_mult = []
best_arvserr_mult = []



for i in range(k):
    best_pnts_all.append(res_f_l3[idx_all[i]])
    best_all_mult.append(res_x_l3[idx_all[i]])
    best_pnts_pdpvserr.append(res_f_l3[idx_pdpvserr[i]])
    best_pdpvserr_mult.append(res_x_l3[idx_pdpvserr[i]])
    best_pnts_arvserr.append(res_f_l3[idx_arvserr[i]])
    best_arvserr_mult.append(res_x_l3[idx_arvserr[i]])

best_pnts_all = np.array(best_pnts_all)
best_pnts_pdpvserr = np.array(best_pnts_pdpvserr)
best_pnts_arvserr = np.array(best_pnts_arvserr)

# print(best_all_mult)
# print(best_pdpvserr_mult)
# print(best_arvserr_mult)

plt.rcParams.update({'font.size': 15})
########################### l3 #############################
plt.figure()
plt.subplot(1, 2, 1)
plt.xlabel('PDP (Wsec)')
plt.ylabel('Accuracy  Loss')
plt.scatter(pdp_all_l3*1e-18, erracc_all_l3, s=4, color='grey')
plt.scatter(res_f_l3[:, 0]*1e-18, res_f_l3[:, 2], s=20, color='red')
plt.scatter(best_pnts_all[:, 0]*1e-18, best_pnts_all[:, 2], s=30, color='green')
plt.scatter(best_pnts_pdpvserr[:, 0]*1e-18, best_pnts_pdpvserr[:, 2], s=30, color='blue')
plt.scatter(best_pnts_arvserr[:, 0]*1e-18, best_pnts_arvserr[:, 2], s=30, color='orange')
plt.rcParams.update({'font.size': 12})
plt.legend(['Exhaustive Solution', 'RNSGA2 Solutions', 'Overall Best Solutions', 'PDP vs Accuracy Loss Best Solutions', 'Area vs Accuracy Loss Best Solutions'], loc ="upper right")
plt.rcParams.update({'font.size': 15})
plt.title("PDP vs Accuracy  Loss")

plt.subplot(1, 2, 2)
plt.xlabel('Area (sq.u)')
plt.ylabel('Accuracy  Loss')
plt.scatter(area_all_l3, erracc_all_l3, s=4, color='grey')
plt.scatter(res_f_l3[:, 1], res_f_l3[:, 2], s=20, color='red')
plt.scatter(best_pnts_all[:, 1], best_pnts_all[:, 2], s=30, color='green')
plt.scatter(best_pnts_pdpvserr[:, 1], best_pnts_pdpvserr[:, 2], s=30, color='blue')
plt.scatter(best_pnts_arvserr[:, 1], best_pnts_arvserr[:, 2], s=30, color='orange')
plt.rcParams.update({'font.size': 12})
plt.legend(['Exhaustive Solution', 'RNSGA2 Solution', 'Overall Best Solutions', 'PDP vs Accuracy Loss Best Solutions', 'Area vs Accuracy Loss Best Solutions'], loc ="upper left")
plt.rcParams.update({'font.size': 15})
plt.title("Area vs Accuracy  Loss")
# plt.suptitle('For 4 Convolutional Layer Network')

plt.rcParams.update({'font.size': 10})
plt.figure()
ax = plt.axes(projection ="3d")
ax.grid(visible= True, color ='grey', linestyle ='-.', linewidth = 0.3, alpha = 0.2)
ax.set_xlabel('PDP (Wsec)', fontweight ='bold')
ax.set_ylabel('Area (sq.u)', fontweight ='bold')
ax.set_zlabel('Accuracy  Loss', fontweight ='bold')
ax.set_zlim([-0.003, 0.003])
ax.scatter3D(all_f_l3[:, 0]*1e-18, all_f_l3[:, 1], all_f_l3[:, 2], s=1, color='grey', depthshade=True, alpha=0.2)
ax.scatter3D(res_f_l3[:, 0]*1e-18, res_f_l3[:, 1], res_f_l3[:, 2], s=20, color = "red", depthshade=False)
ax.scatter3D(best_pnts_all[:, 0]*1e-18, best_pnts_all[:, 1], best_pnts_all[:, 2], s=40, color = "green")                                                                                                                                                                                                                                                                                    
ax.scatter3D(best_pnts_pdpvserr[:, 0]*1e-18, best_pnts_pdpvserr[:, 1], best_pnts_pdpvserr[:, 2], s=40, color='blue')
ax.scatter3D(best_pnts_arvserr[:, 0]*1e-18, best_pnts_arvserr[:, 1], best_pnts_arvserr[:, 2], s=40, color='orange')
plt.rcParams.update({'font.size': 12})
plt.legend(['Exhaustive Solution', 'RNSGA2 Solution', 'Overall Best Solutions', 'PDP vs Accuracy Loss Best Solutions', 'Area vs Accuracy Loss Best Solutions'], loc ="upper right")
plt.rcParams.update({'font.size': 15})
plt.title("For 5 Convolutional Layer Network")

# ########################### l4 #############################
# plt.rcParams.update({'font.size': 15})
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.xlabel('PDP (Wsec)')
# plt.ylabel('Accuracy  Loss')
# plt.scatter(pdp_all_l4*1e-18, erracc_all_l4, s=4, color='grey')
# plt.scatter(res_f_l4[:, 0]*1e-18, res_f_l4[:, 2], s=20, color='red')
# plt.legend(['Exhaustive Solution', 'Optimal Solution'], loc ="upper right")
# plt.title("PDP vs Accuracy  Loss")

# plt.subplot(1, 2, 2)
# plt.xlabel('Area (sq.u)')
# plt.ylabel('Accuracy  Loss')
# plt.scatter(area_all_l4, erracc_all_l4, s=4, color='grey')
# plt.scatter(res_f_l4[:, 1], res_f_l4[:, 2], s=20, color='red')
# plt.legend(['Exhaustive Solution', 'Optimal Solution'], loc ="upper left")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
# plt.title("Area vs Accuracy  Loss")
# plt.suptitle('For 4 Convolutional Layer Network')

# plt.rcParams.update({'font.size': 10})
# plt.figure()
# ax = plt.axes(projection ="3d")
# ax.grid(visible= True, color ='grey', linestyle ='-.', linewidth = 0.3, alpha = 0.2)
# ax.set_xlabel('PDP (Wsec)', fontweight ='bold')
# ax.set_ylabel('Area (sq.u)', fontweight ='bold')
# ax.set_zlabel('Accuracy  Loss', fontweight ='bold')
# ax.scatter3D(all_f_l4[:, 0]*1e-18, all_f_l4[:, 1], all_f_l4[:, 2], s=4, color='grey')
# ax.scatter3D(res_f_l4[:, 0]*1e-18, res_f_l4[:, 1], res_f_l4[:, 2], s=20, color = "red")
# plt.legend(['Exhaustive Solution', 'Optimal Solution'], loc ="upper right")
# plt.suptitle('For 4 Convolutional Layer Network')

# ########################### l5 #############################
# plt.rcParams.update({'font.size': 15})
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.xlabel('PDP (Wsec)')
# plt.ylabel('Accuracy  Loss')
# plt.scatter(pdp_all_l5*1e-18, erracc_all_l5, s=4, color='grey')
# plt.scatter(res_f_l5[:, 0]*1e-18, res_f_l5[:, 2], s=20, color='red')
# plt.legend(['Exhaustive Solution', 'Optimal Solution'], loc ="upper right")
# plt.title("PDP vs Accuracy  Loss")

# plt.subplot(1, 2, 2)
# plt.xlabel('Area (sq.u)')
# plt.ylabel('Accuracy  Loss')
# plt.scatter(area_all_l5, erracc_all_l5, s=4, color='grey')
# plt.scatter(res_f_l5[:, 1], res_f_l5[:, 2], s=20, color='red')
# plt.legend(['Exhaustive Solution', 'Optimal Solution'], loc ="upper left")
# plt.title("Area vs Accuracy  Loss")
# plt.suptitle('For 5 Convolutional Layer Network')

# plt.rcParams.update({'font.size': 10})
# plt.figure()
# ax = plt.axes(projection ="3d")
# ax.grid(visible= True, color ='grey', linestyle ='-.', linewidth = 0.3, alpha = 0.2)
# ax.set_xlabel('PDP (Wsec)', fontweight ='bold')
# ax.set_ylabel('Area (sq.u)', fontweight ='bold')
# ax.set_zlabel('Accuracy  Loss', fontweight ='bold')
# ax.set_zlim([-0.003, 0.003])
# ax.scatter3D(all_f_l5[:, 0]*1e-18, all_f_l5[:, 1], all_f_l5[:, 2], s=1, color='grey', depthshade=True)
# ax.scatter3D(res_f_l5[:, 0]*1e-18, res_f_l5[:, 1], res_f_l5[:, 2], s=20, color = "red", depthshade=False)
# plt.legend(['Exhaustive Solution', 'Optimal Solution'], loc ="upper right")
# plt.suptitle('For 5 Convolutional Layer Network')

plt.show()
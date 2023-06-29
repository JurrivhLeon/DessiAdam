"""
Neural Network and Deep Learning, Final project.
Optimization.
Junyi Liao, 20307110289
VGG Visualization Module.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import os.path as osp


ckpt_dirs = ['./checkpoints/LeNet_SGD', './checkpoints/LeNet_DessiLBI']
epochs = 100
losses = []
trainAcc = []
valAcc = []
for i in range(2):
    loss = np.load(osp.join(ckpt_dirs[i], 'losses.npy'))[:75000].reshape(-1, 100).mean(axis=1)
    losses.append(loss)
    trainAcc.append(np.load(osp.join(ckpt_dirs[i], 'acc.npy'))[0, :])
    valAcc.append(np.load(osp.join(ckpt_dirs[i], 'acc.npy'))[1, :])
losses = np.array(losses)
trainAcc = np.array(trainAcc)
valAcc = np.array(valAcc)
print(valAcc.max(axis=1))

plt.figure()
plt.plot(np.arange(0, losses.shape[1]) * 100, losses.T, linestyle='solid', linewidth=1.2)
plt.xlabel('Steps')
plt.yticks(np.arange(0, 2.0, 0.25))
plt.legend([r'SGD',  r'DessiLBI'])
plt.title('The Loss of LeNet', fontsize=12, pad=10)
plt.show()

filter_norm = np.load(osp.join(ckpt_dirs[0], 'filter_norm.npy'))
sparsity = np.sum(filter_norm[-1, ] < 1e-6)
plt.figure()
plt.plot(filter_norm, linestyle='solid', linewidth=0.75)
plt.xlabel('Epochs')
plt.ylabel('Magnitude')
plt.yticks(np.arange(0, 1.80, 0.20))
plt.title(f'LeNet trained by SGD, Sparsity: ({sparsity}/120)', fontsize=12, pad=10)
plt.show()

filter_norm = np.load(osp.join(ckpt_dirs[1], 'filter_norm.npy'))
sparsity = np.sum(filter_norm[-1, ] < 1e-2)
plt.figure()
plt.plot(filter_norm, linestyle='solid', linewidth=0.75)
plt.xlabel('Epochs')
plt.ylabel('Magnitude')
plt.yticks(np.arange(0, 8.50, 0.80))
plt.title(f'LeNet trained by DessiLBI, Sparsity: ({sparsity}/120)', fontsize=12, pad=10)
plt.show()


ckpt_dirs = []
for opt in ['SGD', 'Adam', 'DessiLBI', 'DessiAdam']:
    ckpt_dirs.append(f'./checkpoints/VGG_A_BN_{opt}')
print(ckpt_dirs)

epochs = 160
losses = []
losses_CI = []
trainAcc = []
valAcc = []

for i in range(4):
    loss = np.load(osp.join(ckpt_dirs[i], 'losses.npy')).reshape(epochs, -1)
    losses.append(loss.mean(axis=-1))
    losses_CI.append(np.quantile(loss, (0.025, 0.975), axis=-1))
    trainAcc.append(np.load(osp.join(ckpt_dirs[i], 'acc.npy'))[0, :])
    valAcc.append(np.load(osp.join(ckpt_dirs[i], 'acc.npy'))[1, :])
trainAcc = np.array(trainAcc)
valAcc = np.array(valAcc)
print(valAcc.max(axis=1))

plt.figure()
plt.plot(np.arange(0, epochs), losses[0],
         linestyle='solid', color='salmon', linewidth=1.2)
plt.plot(np.arange(0, epochs), losses[1],
         linestyle='solid', color='darkmagenta', linewidth=1.2)
plt.plot(np.arange(0, epochs), losses[2],
         linestyle='solid', color='#0057B7', linewidth=1.2)
plt.plot(np.arange(0, epochs), losses[3],
         linestyle='solid', color='#FFD700', linewidth=1.2)
plt.fill_between(np.arange(0, epochs), losses_CI[0][0], losses_CI[0][1],
                 color='lightsalmon', alpha=0.5)
plt.fill_between(np.arange(0, epochs), losses_CI[1][0], losses_CI[1][1],
                 color='violet', alpha=0.5)
plt.fill_between(np.arange(0, epochs), losses_CI[2][0], losses_CI[2][1],
                 color='lightskyblue', alpha=0.5)
plt.fill_between(np.arange(0, epochs), losses_CI[3][0], losses_CI[3][1],
                 color='khaki', alpha=0.5)
plt.xlabel('Epochs')
plt.yscale('log')
plt.legend([r'SGD', r'Adam', r'DessiLBI', r'DessiAdam'])
plt.title('The Loss of VGG-A with BatchNorm', fontsize=12, pad=10)
# plt.grid()
plt.show()

plt.figure()
plt.plot(trainAcc[0], linestyle='solid', color='coral', linewidth=1.2)
plt.plot(trainAcc[1], linestyle='solid', color='darkmagenta', linewidth=1.2)
plt.plot(trainAcc[2], linestyle='solid', color='#0057B7', linewidth=1.2)
plt.plot(trainAcc[3], linestyle='solid', color='#FFD700', linewidth=1.2)
plt.yticks(np.arange(40, 105, 5))
plt.xlabel('Epochs')
plt.legend([r'SGD', r'Adam', r'DessiLBI', r'DessiAdam'])
plt.title('The training accuracy of VGG-A with BatchNorm', fontsize=12, pad=10)
# plt.grid()
plt.show()

plt.figure()
plt.plot(valAcc[0], linestyle='solid', color='coral', linewidth=1.2)
plt.plot(valAcc[1], linestyle='solid', color='darkmagenta', linewidth=1.2)
plt.plot(valAcc[2], linestyle='solid', color='#0057B7', linewidth=1.2)
plt.plot(valAcc[3], linestyle='solid', color='#FFD700', linewidth=1.2)
plt.yticks(np.arange(40, 105, 5))
plt.xlabel('Epochs')
plt.legend([r'SGD', r'Adam', r'DessiLBI', r'DessiAdam'])
plt.title('The test accuracy of VGG-A with BatchNorm', fontsize=12, pad=10)
# plt.grid()
plt.show()

filter_norm = np.load(osp.join(ckpt_dirs[0], 'filter_norm.npy'))
sparsity = np.sum(filter_norm[-1, ] < 1e-4)
plt.figure()
plt.plot(filter_norm, linestyle='solid', linewidth=0.75)
plt.xlabel('Epochs')
plt.ylabel('Magnitude')
plt.yticks(np.arange(0, 2.01, 0.20))
plt.title(f'mSGD, Sparsity: ({sparsity}/512)', fontsize=12, pad=10)
# plt.grid()
plt.show()

filter_norm = np.load(osp.join(ckpt_dirs[1], 'filter_norm.npy'))
sparsity = np.sum(filter_norm[-1, ] < 1e-4)
plt.figure()
plt.plot(filter_norm, linestyle='solid', linewidth=0.75)
plt.xlabel('Epochs')
plt.ylabel('Magnitude')
plt.yticks(np.arange(0, 2.01, 0.20))
plt.title(f'Adam, Sparsity: ({sparsity}/512)', fontsize=12, pad=10)
# plt.grid()
plt.show()

filter_norm = np.load(osp.join(ckpt_dirs[2], 'filter_norm.npy'))
sparsity = np.sum(filter_norm[-1, ] < 1e-4)
plt.figure()
plt.plot(filter_norm, linestyle='solid', linewidth=0.75)
plt.xlabel('Epochs')
plt.ylabel('Magnitude')
plt.yticks(np.arange(0, 1.05, 0.10))
plt.title(f'DessiLBI, Sparsity: ({sparsity}/512)', fontsize=12, pad=10)
# plt.grid()
plt.show()

filter_norm = np.load(osp.join(ckpt_dirs[3], 'filter_norm.npy'))
sparsity = np.sum(filter_norm[-1, ] < 1e-4)
plt.figure()
plt.plot(filter_norm, linestyle='solid', linewidth=0.75)
plt.xlabel('Epochs')
plt.ylabel('Magnitude')
plt.yticks(np.arange(0, 0.91, 0.15))
plt.title(f'DessiAdam, Sparsity: ({sparsity}/512)', fontsize=12, pad=10)
# plt.grid()
plt.show()


ckpt_dirs = []
for opt in ['SGD', 'Adam', 'DessiLBI', 'DessiAdam']:
    ckpt_dirs.append(f'./checkpoints_wd/VGG_A_BN_{opt}')
print(ckpt_dirs)

epochs = 160
losses = []
losses_CI = []
trainAcc = []
valAcc = []

for i in range(4):
    loss = np.load(osp.join(ckpt_dirs[i], 'losses.npy')).reshape(epochs, -1)
    losses.append(loss.mean(axis=-1))
    losses_CI.append(np.quantile(loss, (0.025, 0.975), axis=-1))
    trainAcc.append(np.load(osp.join(ckpt_dirs[i], 'acc.npy'))[0, :])
    valAcc.append(np.load(osp.join(ckpt_dirs[i], 'acc.npy'))[1, :])
trainAcc = np.array(trainAcc)
valAcc = np.array(valAcc)
print(valAcc.max(axis=1))

plt.figure()
plt.plot(np.arange(0, epochs), losses[0],
         linestyle='solid', color='salmon', linewidth=1.2)
plt.plot(np.arange(0, epochs), losses[1],
         linestyle='solid', color='darkmagenta', linewidth=1.2)
plt.plot(np.arange(0, epochs), losses[2],
         linestyle='solid', color='#0057B7', linewidth=1.2)
plt.plot(np.arange(0, epochs), losses[3],
         linestyle='solid', color='#FFD700', linewidth=1.2)
plt.fill_between(np.arange(0, epochs), losses_CI[0][0], losses_CI[0][1],
                 color='lightsalmon', alpha=0.5)
plt.fill_between(np.arange(0, epochs), losses_CI[1][0], losses_CI[1][1],
                 color='violet', alpha=0.5)
plt.fill_between(np.arange(0, epochs), losses_CI[2][0], losses_CI[2][1],
                 color='lightskyblue', alpha=0.5)
plt.fill_between(np.arange(0, epochs), losses_CI[3][0], losses_CI[3][1],
                 color='khaki', alpha=0.5)
plt.xlabel('Epochs')
plt.yscale('log')
plt.legend([r'SGD', r'Adam', r'DessiLBI', r'DessiAdam'])
plt.title('The Loss of VGG-A with BatchNorm and Weight Decay', fontsize=12, pad=10)
# plt.grid()
plt.show()

plt.figure()
plt.plot(trainAcc[0], linestyle='solid', color='coral', linewidth=1.2)
plt.plot(trainAcc[1], linestyle='solid', color='darkmagenta', linewidth=1.2)
plt.plot(trainAcc[2], linestyle='solid', color='#0057B7', linewidth=1.2)
plt.plot(trainAcc[3], linestyle='solid', color='#FFD700', linewidth=1.2)
plt.yticks(np.arange(40, 105, 5))
plt.xlabel('Epochs')
plt.legend([r'SGD', r'Adam', r'DessiLBI', r'DessiAdam'])
plt.title('The training accuracy of VGG-A with BatchNorm and Weight Decay', fontsize=12, pad=10)
# plt.grid()
plt.show()

plt.figure()
plt.plot(valAcc[0], linestyle='solid', color='coral', linewidth=1.2)
plt.plot(valAcc[1], linestyle='solid', color='darkmagenta', linewidth=1.2)
plt.plot(valAcc[2], linestyle='solid', color='#0057B7', linewidth=1.2)
plt.plot(valAcc[3], linestyle='solid', color='#FFD700', linewidth=1.2)
plt.yticks(np.arange(40, 105, 5))
plt.xlabel('Epochs')
plt.legend([r'SGD', r'Adam', r'DessiLBI', r'DessiAdam'])
plt.title('The test accuracy of VGG-A with BatchNorm and Weight Decay', fontsize=12, pad=10)
# plt.grid()
plt.show()

filter_norm = np.load(osp.join(ckpt_dirs[0], 'filter_norm.npy'))
sparsity = np.sum(filter_norm[-1, ] < 1e-4)
plt.figure()
plt.plot(filter_norm, linestyle='solid', linewidth=0.75)
plt.xlabel('Epochs')
plt.ylabel('Magnitude')
plt.yticks(np.arange(0, 1.05, 0.10))
plt.title(f'mSGD, Sparsity: ({sparsity}/512)', fontsize=12, pad=10)
# plt.grid()
plt.show()

filter_norm = np.load(osp.join(ckpt_dirs[1], 'filter_norm.npy'))
sparsity = np.sum(filter_norm[-1, ] < 1e-4)
plt.figure()
plt.plot(filter_norm, linestyle='solid', linewidth=0.75)
plt.xlabel('Epochs')
plt.ylabel('Magnitude')
plt.yticks(np.arange(0, 1.05, 0.10))
plt.title(f'Adam, Sparsity: ({sparsity}/512)', fontsize=12, pad=10)
# plt.grid()
plt.show()

filter_norm = np.load(osp.join(ckpt_dirs[2], 'filter_norm.npy'))
sparsity = np.sum(filter_norm[-1, ] < 1e-4)
plt.figure()
plt.plot(filter_norm, linestyle='solid', linewidth=0.75)
plt.xlabel('Epochs')
plt.ylabel('Magnitude')
plt.yticks(np.arange(0, 1.05, 0.10))
plt.title(f'DessiLBI, Sparsity: ({sparsity}/512)', fontsize=12, pad=10)
# plt.grid()
plt.show()

filter_norm = np.load(osp.join(ckpt_dirs[3], 'filter_norm.npy'))
sparsity = np.sum(filter_norm[-1, ] < 1e-4)
plt.figure()
plt.plot(filter_norm, linestyle='solid', linewidth=0.75)
plt.xlabel('Epochs')
plt.ylabel('Magnitude')
plt.yticks(np.arange(0, 0.81, 0.10))
plt.title(f'DessiAdam, Sparsity: ({sparsity}/512)', fontsize=12, pad=10)
# plt.grid()
plt.show()
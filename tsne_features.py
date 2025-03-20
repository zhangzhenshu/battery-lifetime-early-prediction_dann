import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

feature_s = np.load('epoch_1_source.npy')
feature_t = np.load('epoch_1_target.npy')

# print(feature_t.shape)
feature_s= feature_s.reshape(feature_s.shape[0],-1)
feature_t= feature_t.reshape(feature_t.shape[0],-1)
print(feature_s.shape)
print(feature_t.shape)
features = np.concatenate([feature_s,feature_t])
labels= np.zeros(features.shape[0])
labels[feature_s.shape[0]:]=1
# print(labels)
# print(feature.shape)
tsne = TSNE(random_state=42)
X_tsne = tsne.fit_transform(features)
print(X_tsne.shape)

colors = ['r', 'b']

# plt.figure(figsize=(7, 6), dpi=300)
plt.xlim(X_tsne[:, 0].min()-20, X_tsne[:, 0].max() + 20)
plt.ylim(X_tsne[:, 1].min()-20, X_tsne[:, 1].max() + 20)
# # for i in range(X.shape[0]):
# #     plt.text(X_tsne[i, 0], X_tsne[i, 1], str(int(y[i])),
# #              color=colors[int(y[i])],
# #              fontdict={'weight': 'bold', 'size': 9})
for i in range(features.shape[0]):
    plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color=colors[int(labels[i])], alpha=0.7, s=1)
    # print('\r {} / {}'.format(i, sample_number), end="")

# # title = 'n_sample:' + str(sample_number)
# # plt.title(title)
# plt.legend(['R', 'FI', 'FS', 'DT', 'WI', 'WO'])
# plt.xlabel('Dimension 0')
# plt.ylabel('Dimension 1')
plt.show()
# print('\nTime Usage: {:.2f}'.format(time.time() - start_time))

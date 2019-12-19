import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd

# dictpath = './final_dict_3_123_'
# dictpath_clusgan = '../clusterGAN-master/final_dict_3_123_'
# # supervision_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# supervision_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
# seeds = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]

# manually_extracted = {}

# manually_extracted[0.4] = {}
# manually_extracted[0.8] = {}
# manually_extracted[0.9] = {}
# manually_extracted[0.99] = {}

# manually_extracted[0.4]['acc'] = np.array([0.989, 0.989333, 0.989, 0.989333, 0.989, 0.989333, 0.989, 0.989333, 0.989, 0.989333])
# manually_extracted[0.8]['acc'] = np.zeros([len(seeds)]).astype('float32') + 0.9966666666666667
# manually_extracted[0.9]['acc'] = np.zeros([len(seeds)]).astype('float32') + 0.9993333333333333
# manually_extracted[0.99]['acc'] = np.zeros([len(seeds)]).astype('float32') + 0.9996666666666667

# manually_extracted[0.4]['nmi'] = np.array([0.938690757580867, 0.9401009133560176, 0.938690757580867, 0.9401009133560176, 0.938690757580867, 0.9401009133560176, 0.938690757580867, 0.9401009133560176, 0.938690757580867, 0.9401009133560176])
# manually_extracted[0.8]['nmi'] = np.zeros([len(seeds)]).astype('float32') + 0.9792721325313618
# manually_extracted[0.9]['nmi'] = np.zeros([len(seeds)]).astype('float32') + 0.9952015183800887
# manually_extracted[0.99]['nmi'] = np.zeros([len(seeds)]).astype('float32') + 0.9976006833367572

# manually_extracted[0.4]['ari'] = np.array([0.967262837315131, 0.9682444338561488, 0.967262837315131, 0.9682444338561488, 0.967262837315131, 0.9682444338561488, 0.967262837315131, 0.9682444338561488, 0.967262837315131, 0.9682444338561488])
# manually_extracted[0.8]['ari'] = np.zeros([len(seeds)]).astype('float32') + 0.9900238919960412
# manually_extracted[0.9]['ari'] = np.zeros([len(seeds)]).astype('float32') + 0.9980001666669726
# manually_extracted[0.99]['ari'] = np.zeros([len(seeds)]).astype('float32') + 0.9989998331666945

# My_Acc = np.zeros([len(seeds), len(supervision_levels)]).astype('float32')
# My_ARI = np.zeros([len(seeds), len(supervision_levels)]).astype('float32')
# My_NMI = np.zeros([len(seeds), len(supervision_levels)]).astype('float32')

# sup_no = 0
# for supervision_level in supervision_levels:
# 	seed_no = 0
# 	if supervision_level in [0.4, 0.8, 0.9, 0.99]:
# 		My_Acc[:, sup_no] = manually_extracted[supervision_level]['acc']
# 		My_ARI[:, sup_no] = manually_extracted[supervision_level]['acc']
# 		My_NMI[:, sup_no] = manually_extracted[supervision_level]['nmi']
# 	else:
# 		for seed in seeds:
# 			dict1 = pkl.load(open(dictpath + str(supervision_level) + '.pkl', "rb"), encoding='latin1')
# 			My_Acc[seed_no][sup_no] = dict1[seed]['acc']
# 			My_ARI[seed_no][sup_no] = dict1[seed]['ari']
# 			My_NMI[seed_no][sup_no] = dict1[seed]['nmi']
# 			seed_no += 1
# 	sup_no += 1


# seeds = [0., 1., 2., 3., 4., 5.]
# Clusgan_Acc = np.zeros([len(seeds), len(supervision_levels)]).astype('float32')
# Clusgan_ARI = np.zeros([len(seeds), len(supervision_levels)]).astype('float32')
# Clusgan_NMI = np.zeros([len(seeds), len(supervision_levels)]).astype('float32')

# sup_no = 0
# for supervision_level in supervision_levels:
# 	seed_no = 0
# 	for seed in seeds:
# 		dict1 = pkl.load(open(dictpath_clusgan + str(supervision_level) + '.pkl', "rb"), encoding='latin1')
# 		# print(dict1.keys())
# 		Clusgan_Acc[seed_no][sup_no] = dict1[seed]['acc']
# 		Clusgan_ARI[seed_no][sup_no] = dict1[seed]['ari']
# 		Clusgan_NMI[seed_no][sup_no] = dict1[seed]['nmi']
# 		seed_no += 1
# 	sup_no += 1


# sp_ACC = [0.9316666666666666, 0.9333333333333333, 0.931, 0.9333333333333333, 0.9306666666666666, \
# 0.938, 0.9336666666666666, 0.934, 0.934, 0.9336666666666666]
# sp_NMI = [0.7792540455457132, 0.7789347282483188, 0.7796610129712894, 0.7897724641491273, \
# 0.7806444773137634, 0.7954069310573736, 0.7847985568136808, 0.7909123385026358, \
# 0.7894751144079943, 0.7889006122566861]
# sp_ARI = [0.8101270596218558, 0.8140813553409595, 0.8086501590042656, 0.815065000634403, \
# 0.8078622925802322, 0.8265887499981893, 0.8153754048443241, 0.8167088666659972, \
# 0.8165601558818674, 0.8157363571886375]

# our_mix_gans = np.array([0.964, 0.971, 0.97334, 0.97766, 0.98233, 0.9833, 0.98566, 0.98733, 0.99233, 0.99766,  1.0])

# My_Acc[:, 0] = sp_ACC
# My_ARI[:, 0] = sp_ARI
# My_NMI[:, 0] = sp_NMI


# fig = plt.figure()
# ax = fig.add_subplot(111)
# y_spacing = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# My_Acc = np.sort(My_Acc, axis=0)
# Clusgan_Acc = np.sort(Clusgan_Acc, axis=0)

# yerror1 = np.zeros([2, My_Acc.shape[1]]).astype('float32')
# yerror1[0] = My_Acc[-1, :] - My_Acc[int(My_Acc.shape[0] / 2), :]
# yerror1[1] = My_Acc[int(My_Acc.shape[0] / 2), :] - My_Acc[0, :]

# yerror2 = np.zeros([2, Clusgan_Acc.shape[1]]).astype('float32')
# yerror2[1] = Clusgan_Acc[int(Clusgan_Acc.shape[0] / 2), :] - Clusgan_Acc[1, :]
# yerror2[0] = Clusgan_Acc[-1, :] - Clusgan_Acc[int(Clusgan_Acc.shape[0] / 2), :]

# ax.errorbar(supervision_levels[:], My_Acc[int(My_Acc.shape[0] / 2), :], yerror1, label="Our Method*")
# ax.errorbar(supervision_levels[:], Clusgan_Acc[int(Clusgan_Acc.shape[0] / 2), :], yerror2, label="ClusterGAN")
# ax.plot(supervision_levels[:], our_mix_gans, label="Our Mix-GANs*")


# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
# plt.xlabel('Supervision')
# plt.ylabel('Accuracy')
# plt.title('Median Acc vs Supervision')
# plt.show()

# df = pd.DataFrame(My_Acc[:, 1:], columns=supervision_levels[1:])
# boxplot = df.boxplot(column=supervision_levels[1:])

# df1 = pd.DataFrame(Clusgan_Acc[:, 1:], columns=supervision_levels[1:])
# boxplot1 = df1.boxplot(column=supervision_levels[1:])
# ax.boxplot(My_Acc)
# ax.boxplot(Clusgan_Acc)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# y_spacing = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# My_NMI = np.sort(My_NMI, axis=0)
# Clusgan_NMI = np.sort(Clusgan_NMI, axis=0)

# yerror1 = np.zeros([2, My_NMI.shape[1]]).astype('float32')
# yerror1[0] = My_NMI[-1, :] - My_NMI[int(My_NMI.shape[0] / 2), :]
# yerror1[1] = My_NMI[int(My_NMI.shape[0] / 2), :] - My_NMI[0, :]

# yerror2 = np.zeros([2, Clusgan_NMI.shape[1]]).astype('float32')
# yerror2[1] = Clusgan_NMI[int(Clusgan_NMI.shape[0] / 2), :] - Clusgan_NMI[1, :]
# yerror2[0] = Clusgan_NMI[-1, :] - Clusgan_NMI[int(Clusgan_NMI.shape[0] / 2), :]

# ax.errorbar(supervision_levels[:], My_NMI[int(My_NMI.shape[0] / 2), :], yerror1, label="Our Method*")
# ax.errorbar(supervision_levels[:], Clusgan_NMI[int(Clusgan_NMI.shape[0] / 2), :], yerror2, label="ClusterGAN")

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
# plt.xlabel('Supervision')
# plt.ylabel('NMI')
# plt.title('Median NMI vs Supervision')
# plt.show()


# plt.boxplot(My_Ari, positions=y_spacing)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# y_spacing = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# My_ARI = np.sort(My_ARI, axis=0)
# Clusgan_ARI = np.sort(Clusgan_ARI, axis=0)

# yerror1 = np.zeros([2, My_ARI.shape[1]]).astype('float32')
# yerror1[0] = My_ARI[-1, :] - My_ARI[int(My_ARI.shape[0] / 2), :]
# yerror1[1] = My_ARI[int(My_ARI.shape[0] / 2), :] - My_ARI[0, :]

# yerror2 = np.zeros([2, Clusgan_ARI.shape[1]]).astype('float32')
# yerror2[1] = Clusgan_ARI[int(Clusgan_ARI.shape[0] / 2), :] - Clusgan_ARI[1, :]
# yerror2[0] = Clusgan_ARI[-1, :] - Clusgan_ARI[int(Clusgan_ARI.shape[0] / 2), :]

# ax.errorbar(supervision_levels[:], My_ARI[int(My_ARI.shape[0] / 2), :], yerror1, label="Our Method*")
# ax.errorbar(supervision_levels[:], Clusgan_ARI[int(Clusgan_ARI.shape[0] / 2), :], yerror2, label="ClusterGAN")
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
# plt.xlabel('Supervision')
# plt.ylabel('ARI')
# plt.title('Median ARI vs Supervision')
# plt.show()

# plt.boxplot(My_NMI, positions=y_spacing)
# plt.show()



##############################
# 10 digits
##############################

all_Acc = np.array([[0.9603, 0.9706, 0.98, 0.9831, 0.9872, 0.9899, 0.994, 0.996, 0.9979, 0.9993],\
 [0.9602, 0.9706, 0.9802, 0.9831, 0.9873, 0.9899, 0.994, 0.996, 0.9979, 0.9993],\
  [0.9598, 0.9706, 0.9801, 0.9831, 0.9871, 0.9898, 0.9939, 0.9958, 0.9978, 0.9993]])

all_Ari = np.array([[0.9140761064283607, 0.9359533966272853, 0.9561727839861769, 0.9628664160358757, 0.971792965714945, 0.9777028708164567, 0.9867160301485769, 0.991138165390257, 0.995337367887063, 0.9984449343255523],\
 [0.9138803623310873, 0.9359533966272853, 0.9566034896392049, 0.9628664160358757, 0.972015037506719, 0.9777028708164567, 0.9867160301485769, 0.991138165390257, 0.995337367887063, 0.9984449343255523],\
  [0.9130778078998264, 0.9359533966272853, 0.9563893793722652, 0.9628664160358757, 0.9715729944414241, 0.9774843254374449, 0.9864947238764955, 0.9906966534466745, 0.9951155006510589, 0.9984449343255523]])

all_NMI = np.array([[0.9018072608085563, 0.9246421439338726, 0.9458619247686076, 0.9532356500468816, 0.9630071924290071, 0.9700884014368077, 0.9813973558421007, 0.9873737076222832, 0.9930297816157724, 0.9977165360172092],\
 [0.9018994093456868, 0.9246421439338726, 0.9464484100973412, 0.9532356500468816, 0.9633505833132529, 0.9700884014368077, 0.9813973558421007, 0.9873737076222832, 0.9930297816157724, 0.9977165360172092],\
  [0.900847266219067, 0.9246421439338726, 0.9458619247686076, 0.9532356500468816, 0.9626642196898674, 0.9698058820247453, 0.981054121200665, 0.9867476218936879, 0.9926864393462932, 0.9977165360172092]])


clusgan_all_Acc = np.array([[0.4573, 0.8872, 0.9225, 0.944, 0.9634, 0.9724, 0.9795, 0.9823, 0.9903, 0.9944, 0.9996],\
 [0.4586, 0.8838, 0.9225, 0.9504, 0.9633, 0.9753, 0.9816, 0.9845, 0.9895, 0.9947, 0.9994],\
  [0.4677, 0.8923, 0.9216, 0.944, 0.9612, 0.9725, 0.9787, 0.9784, 0.9887, 0.9947, 0.9993]])

clusgan_all_Ari = np.array([[0.23370211440933777, 0.7676097645731287, 0.835950523885825, 0.8799489989903572, 0.9205810428989428, 0.9396735733354892, 0.9549907872790253, 0.9610640764413785, 0.9785536948391552, 0.9875958519685046, 0.9991107556299529],\
 [0.23715803246228156, 0.761422765782493, 0.835950523885825, 0.8930743291884765, 0.9203044238226911, 0.9459951948282982, 0.9595618838634654, 0.9658687963580472, 0.9768106032723688, 0.9882587364965643, 0.9986663565793101],\
  [0.23349543267698172, 0.7771879187695593, 0.8343989047510579, 0.8799489989903572, 0.9158865432305331, 0.93991578216311, 0.9533087897564144, 0.952609241989133, 0.9750272038689127, 0.9882528329355562, 0.9984447116264218]])

clusgan_all_NMI = np.array([[0.30970375733382355, 0.7606690749337938, 0.8241836312668102, 0.8648672861054831, 0.9058868960817973, 0.9256260058389296, 0.9419513886734417, 0.9486585171873599, 0.9699534981296588, 0.9823743337782944, 0.9986262821006503],\
 [0.3369255045857179, 0.7587236735635194, 0.8241836312668102, 0.8748027777383911, 0.9042157674618122, 0.9331901701663551, 0.9469181755588725, 0.9541249314920722, 0.9682548332843991, 0.9833435963792218, 0.9979394663988271],\
  [0.32847447666503404, 0.7728963834740866, 0.8220877628885804, 0.8648672861054831, 0.9004555629107841, 0.9256790186607428, 0.9411604741625772, 0.9392052201187494, 0.9658445795794869, 0.9828357597717291, 0.9975961675615096]])


# #####################
supervision_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

# print(all_Acc.shape)
# print(clusgan_all_Acc.shape)


# fig = plt.figure()
# ax = fig.add_subplot(111)
# y_spacing = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# all_Acc = np.sort(all_Acc, axis=0)
# clusgan_all_Acc = np.sort(clusgan_all_Acc, axis=0)

# yerror1 = np.zeros([2, all_Acc.shape[1]]).astype('float32')
# yerror1[0] = all_Acc[-1, :] - all_Acc[int(all_Acc.shape[0] / 2), :]
# yerror1[1] = all_Acc[int(all_Acc.shape[0] / 2), :] - all_Acc[0, :]

# yerror2 = np.zeros([2, clusgan_all_Acc.shape[1]]).astype('float32')
# yerror2[1] = clusgan_all_Acc[int(clusgan_all_Acc.shape[0] / 2), :] - clusgan_all_Acc[1, :]
# yerror2[0] = clusgan_all_Acc[-1, :] - clusgan_all_Acc[int(clusgan_all_Acc.shape[0] / 2), :]

# ax.errorbar(supervision_levels[1:], all_Acc[int(all_Acc.shape[0] / 2), :], yerror1, label="Our Method*")
# ax.errorbar(supervision_levels[1:], clusgan_all_Acc[int(clusgan_all_Acc.shape[0] / 2), 1:], yerror2[:, 1:], label="ClusterGAN")
# # ax.plot(supervision_levels[:], our_mix_gans, label="Our Mix-GANs*")

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
# plt.xlabel('Supervision')
# plt.ylabel('Accuracy')
# plt.title('Median Acc vs Supervision')
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111)
# y_spacing = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# all_Ari = np.sort(all_Ari, axis=0)
# clusgan_all_Ari = np.sort(clusgan_all_Ari, axis=0)

# yerror1 = np.zeros([2, all_Ari.shape[1]]).astype('float32')
# yerror1[0] = all_Ari[-1, :] - all_Ari[int(all_Ari.shape[0] / 2), :]
# yerror1[1] = all_Ari[int(all_Ari.shape[0] / 2), :] - all_Ari[0, :]

# yerror2 = np.zeros([2, clusgan_all_Ari.shape[1]]).astype('float32')
# yerror2[1] = clusgan_all_Ari[int(clusgan_all_Ari.shape[0] / 2), :] - clusgan_all_Ari[1, :]
# yerror2[0] = clusgan_all_Ari[-1, :] - clusgan_all_Ari[int(clusgan_all_Ari.shape[0] / 2), :]

# ax.errorbar(supervision_levels[1:], all_Ari[int(all_Ari.shape[0] / 2), :], yerror1, label="Our Method*")
# ax.errorbar(supervision_levels[1:], clusgan_all_Ari[int(clusgan_all_Ari.shape[0] / 2), 1:], yerror2[:, 1:], label="ClusterGAN")
# # ax.plot(supervision_levels[:], our_mix_gans, label="Our Mix-GANs*")

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
# plt.xlabel('Supervision')
# plt.ylabel('ARI')
# plt.title('Median ARI vs Supervision')
# plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
y_spacing = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
all_NMI = np.sort(all_NMI, axis=0)
clusgan_all_NMI = np.sort(clusgan_all_NMI, axis=0)

yerror1 = np.zeros([2, all_NMI.shape[1]]).astype('float32')
yerror1[0] = all_NMI[-1, :] - all_NMI[int(all_NMI.shape[0] / 2), :]
yerror1[1] = all_NMI[int(all_NMI.shape[0] / 2), :] - all_NMI[0, :]

yerror2 = np.zeros([2, clusgan_all_NMI.shape[1]]).astype('float32')
yerror2[1] = clusgan_all_NMI[int(clusgan_all_NMI.shape[0] / 2), :] - clusgan_all_NMI[1, :]
yerror2[0] = clusgan_all_NMI[-1, :] - clusgan_all_NMI[int(clusgan_all_NMI.shape[0] / 2), :]

ax.errorbar(supervision_levels[1:], all_NMI[int(all_NMI.shape[0] / 2), :], yerror1, label="Our Method*")
ax.errorbar(supervision_levels[1:], clusgan_all_NMI[int(clusgan_all_NMI.shape[0] / 2), 1:], yerror2[:, 1:], label="ClusterGAN")
# ax.plot(supervision_levels[:], our_mix_gans, label="Our Mix-GANs*")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.xlabel('Supervision')
plt.ylabel('NMI')
plt.title('Median NMI vs Supervision')
plt.show()
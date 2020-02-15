import FCN
import imageUnits

batch_size = 50
channels = 3
nclass = 2

path = 'data/train/*.tif'

data_provider = imageUnits.ImageProvider(path=path, bathsize=batch_size, shuffle_data=True, channels=channels,
                                         nclass=nclass)
net = FCN.FC_net(data_provider, batch_size, conv_size=3, pool_size=2, channels=channels, nclass=nclass,
                 save_path='model',
                 white_channel_weight=0.5,
                 layers=5)
net.trian(epochs=10, train_iters=50, keep_prob=0.5, learn_rate=0.005, restore=False, save_steps=50, loss_name='cross')

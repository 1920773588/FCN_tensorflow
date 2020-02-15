import FCN
import imageUnits

batch_size = 1
channels = 3
nclass = 2

path = 'data/train/*.tif'

data_provider = imageUnits.ImageProvider(path=path, bathsize=batch_size, shuffle_data=True, channels=channels,
                                         nclass=nclass)
net = FCN.FC_net(data_provider, batch_size, conv_size=3, pool_size=2, channels=channels, nclass=nclass,
                 save_path='model',
                 white_channel_weight=0.5,
                 layers=5)
net.predite()

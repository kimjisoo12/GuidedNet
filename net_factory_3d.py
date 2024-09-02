from networks.unet_3D import unet_3D

from networks.unet_3D_gmm import unet_gmm

def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "guidedNet":
        net = unet_gmm(n_classes=class_num, in_channels=in_chns).cuda()
    else:
        net = None
    return net

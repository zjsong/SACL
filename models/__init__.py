"""
Construct network models based on existing network classes.
"""


from .models import ResNet18_, VGGish128


class ModelBuilder():

    def build_frame(self, arch='resnet18', train_from_scratch=False, fine_tune=False, weights_resnet_imgnet=None):

        if arch == 'resnet18':
            if train_from_scratch:
                pretrained = False
                net_frame = ResNet18_(pretrained, modal='vision')
            else:
                pretrained = True
                net_frame = ResNet18_(pretrained, modal='vision', resnet18_imgnet=weights_resnet_imgnet)
                if not fine_tune:
                    for param in net_frame.parameters():
                        param.requires_grad = False

        else:
            raise Exception('Architecture undefined!')

        return net_frame

    def build_sound(self, arch='vggish', weights_vggish=None, out_dim=512):

        if arch == 'vggish':
            net_sound = VGGish128(weights_vggish, out_dim)

            # for p in net_sound.features.parameters():
            #     p.requires_grad = False
            # for p in net_sound.embeddings.parameters():
            #     p.requires_grad = False

        else:
            raise Exception('Architecture undefined!')

        return net_sound

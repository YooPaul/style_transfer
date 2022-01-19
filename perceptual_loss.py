import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptualLoss():
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()

        vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).eval()
        self.vgg16  = vgg16.features.to(device) # Just extract the fully convolutional model

        # placeholder for batch features
        self.features = {}

        # Helper function for feature extraction
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        # Register Hook
        self.layers = [('relu1_2', 3), ('relu2_2', 8), ('relu3_3', 15), ('relu4_3', 22)]
        for layer_name, layer_index in self.layers:
            self.vgg16[layer_index].register_forward_hook(get_features(layer_name))

        '''
        # forward pass [with feature extraction]
        preds = vgg16(torch.rand((1,3,256,256)))

        for layer_name, _ in layers:
            print(features[layer_name].shape)
        '''

    def feature_reconstruction_loss(self, x1, x2):
        x = torch.concat((x1, x2), dim=0)
        self.vgg16(x)
        feature_maps = self.features['relu3_3']

        N = x1.shape[0]
        C, H, W = feature_maps.shape[1:]
        feature_map_x1 = feature_maps[:N]
        feature_map_x2 = feature_maps[N:]
        return F.mse_loss(feature_map_x1, feature_map_x2) # mse_loss already normalizes / (C*H*W)

    def  style_reconstruction_loss(self, x1, x2):
        x = torch.concat((x1, x2), dim=0)
        self.vgg16(x)
        N = x1.shape[0]
        

        loss = 0
        for layer_name, _ in self.layers:
            feature_maps = self.features[layer_name]
            C, H, W = feature_maps.shape[1:]
            
            feature_map_x1 = feature_maps[:N].reshape((N, C, -1))
            feature_map_x2 = feature_maps[N:].reshape((N, C, -1))

            gram_mat_x1 = feature_map_x1 @ torch.transpose(feature_map_x1, -1, -2) / (C*H*W)
            gram_mat_x2 = feature_map_x2 @ torch.transpose(feature_map_x2, -1, -2) / (C*H*W)
            
            #loss += torch.square(torch.norm(gram_mat_x1 - gram_mat_x2, p='fro',dim=(1,2))).mean() #F.mse_loss(feature_map_x1, feature_map_x2)
            loss += torch.square(torch.linalg.matrix_norm(gram_mat_x1 - gram_mat_x2)).mean() # Frobenius norm
        return loss


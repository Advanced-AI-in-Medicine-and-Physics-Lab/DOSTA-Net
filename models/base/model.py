import torch
from . import initialization as init
from torch import nn


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


class SegmentationModelResidual(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels


        n = x.shape[1]
        middle_channel = n // 2  
        return masks+x[:, middle_channel, :, :].unsqueeze(1)
        

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x
       
    

class SegmentationResidual2Input(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def reshape_layer(self, x, features):
        b, n, _, _ = x.shape
        output_features = []
        for i, feat in enumerate(features[:-1]):
            # feat 尺寸为 [b*n, k, w, h]，reshape 回 [b, n, k, w, h]
            _, k, w, h = feat.shape
            feat = feat.view(b, n, k, w, h)
            
            feat = feat.mean(dim=1)
            output_features.append(feat)
        
        return output_features


    def random_channel_swap(self, s: torch.Tensor, r: torch.Tensor, ratio_swap: int):
        """
        :param s: Tensor， [b, n, w, h]
        :param r: Tensor， [b, n, w, h]
        :param num_swap: 
        :return: 
        """
        assert s.shape == r.shape, f"s and r must have the same shape{s.shape} and {r.shape}"
        b, n, w, h = s.shape
        num_swap = max(1, int(n * (ratio_swap / 100)))  # Ensure at least one channel is swapped
        assert 0 < num_swap <= n, "num_swap must be in the range (0, n]"
        
        s_swapped = s.clone()
        r_swapped = r.clone()
        
        for i in range(b):
            idx = torch.randperm(n)[:num_swap]
            
            s_swapped[i, idx] = r[i, idx]
            r_swapped[i, idx] = s[i, idx]
        
        return s_swapped, r_swapped
    
    
    def forward(self, x_s, x_r):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features_s = self.encoder(x_s)
    

        features_r = self.encoder(x_r)

        last_feature_s = features_s[-1]
        last_feature_r = features_r[-1]
        if self.is_mixup:
            last_feature_s, last_feature_r = self.random_channel_swap(last_feature_s, last_feature_r, self.ratio_swap)
        
        last_feature_s = self.attention_layer(last_feature_s)
        last_feature_r = self.attention_layer(last_feature_r)

        bottleneck_features_s = features_s[:-1]+[last_feature_s]
        bottleneck_features_r = features_r[:-1]+[last_feature_r]



        decoder_output_s = self.decoder(*bottleneck_features_s)
        masks_s = self.segmentation_head(decoder_output_s)

        
        decoder_output_r = self.decoder(*bottleneck_features_r)
        masks_r = self.segmentation_head(decoder_output_r)

        n = x_s.shape[1]  
        middle_channel = n // 2  
        
        return masks_s+x_s[:, middle_channel, :, :].unsqueeze(1), masks_r+x_r[:, middle_channel, :, :].unsqueeze(1)

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


class SegmentationResidual2InputV2(torch.nn.Module):
    '''
    The difference between SegmentationResidual2InputV2 and SegmentationResidual2Input is that the former uses the \
    domain-shuffle temporal atention (DSTA) for the feature map of the encoder.
    While the latter uses the attention layer for the every stage feature map of the encoder.
    '''
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def reshape_layer(self, x, features):
        b, n, _, _ = x.shape
        output_features = []
        for i, feat in enumerate(features[:-1]):
            _, k, w, h = feat.shape
            feat = feat.view(b, n, k, w, h)
            
            feat = feat.mean(dim=1)
            output_features.append(feat)
        
        return output_features


    def random_channel_swap(self, s_list: list, r_list: list, N: float):

        assert len(s_list) == len(r_list), "s_list and r_list must have the same length"

        DSTA_swapped_list_s = [s_list[0]]
        DSTA_swapped_list_r = [r_list[0]]
        
        for index, (s, r) in enumerate(zip(s_list[1:], r_list[1:])):
            if self.is_mixup:
                assert s.shape == r.shape, f"Shapes do not match: {s.shape} vs {r.shape}"
                b, n, w, h = s.shape
                assert 0 < N <= 100, "N must be in the range (0, 100]"

                num_swap = max(1, int(n * (N / 100)))  # Ensure at least one channel is swapped

                s_swapped = s.clone()
                r_swapped = r.clone()

                for i in range(b):
                    idx = torch.randperm(n)[:num_swap]

                    s_swapped[i, idx] = r[i, idx]
                    r_swapped[i, idx] = s[i, idx]
            else:
                s_swapped = s
                r_swapped = r
            DSTA_feature_s = self.attention_layers[index](s_swapped)
            DSTA_feature_r = self.attention_layers[index](r_swapped)

            DSTA_swapped_list_s.append(DSTA_feature_s)
            DSTA_swapped_list_r.append(DSTA_feature_r)

        return DSTA_swapped_list_s, DSTA_swapped_list_r

    
    def forward(self, x_s, x_r):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features_s = self.encoder(x_s)
        

        features_r = self.encoder(x_r)

        if self.is_att:
            bottleneck_features_s, bottleneck_features_r = self.random_channel_swap(features_s, features_r, self.ratio_swap)
        else:
            bottleneck_features_s = features_s
            bottleneck_features_r = features_r

        decoder_output_s = self.decoder(*bottleneck_features_s)
        masks_s = self.segmentation_head(decoder_output_s)

        
        decoder_output_r = self.decoder(*bottleneck_features_r)
        masks_r = self.segmentation_head(decoder_output_r)

        n = x_s.shape[1]  
        middle_channel = n // 2 
        
        return masks_s+x_s[:, middle_channel, :, :].unsqueeze(1), masks_r+x_r[:, middle_channel, :, :].unsqueeze(1)

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


class SegmentationModelNoSkip(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

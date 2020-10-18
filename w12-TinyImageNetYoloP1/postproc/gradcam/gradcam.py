import torch
import torch.nn.functional as F


class GradCAM():
    def __init__(self, model, layer_str):
        self.layer_str = layer_str
        self.activations ={}
        self.gradients = {}
        self.model = model

        self.model_layer = {1: model.layer1,
                           2: model.layer2,
                           3: model.layer3,
                           4: model.layer4,
                           }
        self.set_intent_layer()

        def activation_hook(module, input, activation):
            self.activations['value'] = activation

        def gradient_hook(module, input, gradient):
            self.gradients['value'] = gradient[0]

        self.intent_layer.register_forward_hook(activation_hook)
        self.intent_layer.register_backward_hook(gradient_hook)


    def set_intent_layer(self):
        layer_idx = int(str(self.layer_str).lstrip('layer_'))
        print('intent layer',str(layer_idx))
        if layer_idx > 0 and layer_idx < 5:
            self.intent_layer = self.model_layer[layer_idx]

    def saliency_map_size(self, *input_size):
        device = next(self.model.parameters()).device
        self.model(torch.zeros(1, 3, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
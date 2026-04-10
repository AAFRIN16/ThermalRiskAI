import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Hook into last conv layer of EfficientNet-B0
        target_layer = self.model.backbone.features[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot)

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def generate_gradcam_overlay(model, image_pil, device, transform):
    """Generate Grad-CAM overlay and return as base64 encoded image."""
    import base64
    import io

    img_tensor = transform(image_pil).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)

    gradcam = GradCAM(model)
    cam = gradcam.generate(img_tensor)

    # Convert PIL to numpy
    img_np = np.array(image_pil.resize((224, 224)))

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = (0.5 * img_np + 0.5 * heatmap).astype(np.uint8)

    # Encode to base64
    pil_overlay = Image.fromarray(overlay)
    buffer = io.BytesIO()
    pil_overlay.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return b64, cam.tolist()
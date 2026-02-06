import torch
import numpy as np
import matplotlib
from PIL import Image
import os

from unidepth.models import UniDepthV2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEMO_IMG_PATH = os.path.join(SCRIPT_DIR, '..', 'src', 'imgs', 'hand.jpg')



class DepthMarker:
    def __init__(self, type_: str = "l"):
        self.save_dir = os.path.join(SCRIPT_DIR, '..', 'src','imgs')
        os.makedirs(self.save_dir, exist_ok=True)
        name = f"unidepth-v2-vit{type_}14"
        self.device = torch.device("cuda")
        self.model = UniDepthV2.from_pretrained(
            f"lpiccinelli/{name}"
        ).to(self.device).eval()

    def _read_img(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)  
        img_torch = torch.from_numpy(img_np).permute(2, 0, 1).float()
        img_torch /= 255.0
        return img_torch.to(self.device)

    def annotate(self, path: str):
        rgb = self._read_img(path)
        with torch.no_grad():
            preds = self.model.infer(rgb, None)
        return preds
    
    def depth_to_colormap(self,
        depth: torch.Tensor, vmin=None, vmax=None, cmap="magma", inverse: bool = True
    ):
        depth_cpu = depth.detach().cpu().numpy()

        if vmin is None:
            vmin = np.percentile(depth_cpu, 5)
        if vmax is None:
            vmax = np.percentile(depth_cpu, 95)

        depth_norm = np.clip((depth_cpu - vmin) / (vmax - vmin), 0, 1)
        cmap_fn = matplotlib.colormaps.get_cmap(cmap) if not inverse else matplotlib.colormaps.get_cmap(cmap+"_r")

        colored = cmap_fn(depth_norm)[:, :, :3] 

        return (colored * 255).astype(np.uint8)

    def render_depth(self, path: str):
        base_name = os.path.basename(path).split(".")[0]
        to_path = f"{self.save_dir}/{base_name}_unidepth.png"
        preds = self.annotate(path)
        depth = preds["depth"].squeeze()

        depth_color = self.depth_to_colormap(depth, cmap="magma")
        Image.fromarray(depth_color).save(to_path)
        return to_path


def demo():
    annotator = DepthMarker("l")
    to_path = annotator.render_depth(DEFAULT_DEMO_IMG_PATH)
    print("Saved demo to", to_path  )


if __name__ == "__main__":
    demo()
    pass

import torch
import torch.nn as nn
import os

class StupidNet(nn.Module):
    def __init__(self):
        super(StupidNet, self).__init__()

    def forward(self, ram):
        mi_x = ram[:, 32].float()
        mi_y = ram[:, 34].float()
        su_x = ram[:, 33].float()
        su_y = ram[:, 35].float()

        dist_x = torch.abs(su_x - mi_x)
        dist_y = torch.abs(su_y - mi_y)

        # 1. Align Y perfectly (dist_y > 2)
        cond_y = dist_y > 2
        act_y = torch.where(su_y > mi_y, torch.tensor(5), torch.tensor(2))

        # 2. Align X perfectly (dist_x > 24)
        cond_x = dist_x > 24
        act_x = torch.where(su_x > mi_x, torch.tensor(3), torch.tensor(4))
        
        # 3. Back up if too close (dist_x < 22)
        cond_x_close = dist_x < 22
        act_x_close = torch.where(su_x > mi_x, torch.tensor(4), torch.tensor(3))

        action = torch.tensor(1) # Default punch
        action = torch.where(cond_x_close, act_x_close, action)
        action = torch.where(cond_x, act_x, action)
        action = torch.where(cond_y, act_y, action) # Y has priority!

        logits = torch.zeros(1, 6)
        logits.scatter_(1, action.unsqueeze(0), 1.0)
        return logits

def export():
    model = StupidNet()
    model.eval()
    dummy_input = torch.zeros(1, 128, dtype=torch.uint8)
    onnx_path = os.path.join(os.path.dirname(__file__), "stupid.onnx")
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"])

if __name__ == "__main__":
    export()

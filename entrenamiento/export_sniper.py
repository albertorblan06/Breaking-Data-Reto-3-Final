import torch
import torch.nn as nn
import os

class SniperNet(nn.Module):
    def __init__(self):
        super(SniperNet, self).__init__()

    def forward(self, ram):
        mi_x = ram[:, 32].float()
        mi_y = ram[:, 34].float()
        su_x = ram[:, 33].float()
        su_y = ram[:, 35].float()

        dist_x = torch.abs(su_x - mi_x)
        dist_y = torch.abs(su_y - mi_y)

        # 1. Stay exactly 27 pixels away horizontally!
        cond_x_far = dist_x > 27
        cond_x_close = dist_x < 26
        
        act_x_far = torch.where(su_x > mi_x, torch.tensor(3), torch.tensor(4)) # move right if they are right
        act_x_close = torch.where(su_x > mi_x, torch.tensor(4), torch.tensor(3)) # move left if they are right
        
        # 2. Stay exactly 0 pixels away vertically!
        cond_y_misaligned = dist_y > 2
        act_y = torch.where(su_y > mi_y, torch.tensor(5), torch.tensor(2))

        # 3. Punch only if we are exactly in the sweet spot: dist_x is 26 or 27, and Y is aligned
        cond_punch = (dist_x >= 26) & (dist_x <= 27) & (dist_y <= 2)
        act_punch = torch.tensor(1)

        action = torch.tensor(0)
        # Priority: Back up if too close!
        action = torch.where(cond_x_far, act_x_far, action)
        action = torch.where(cond_y_misaligned, act_y, action)
        action = torch.where(cond_x_close, act_x_close, action)
        action = torch.where(cond_punch, act_punch, action)

        logits = torch.zeros(1, 6)
        logits.scatter_(1, action.unsqueeze(0), 1.0)
        return logits

def export():
    model = SniperNet()
    model.eval()
    dummy_input = torch.zeros(1, 128, dtype=torch.uint8)
    onnx_path = os.path.join(os.path.dirname(__file__), "sniper.onnx")
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"])

if __name__ == "__main__":
    export()

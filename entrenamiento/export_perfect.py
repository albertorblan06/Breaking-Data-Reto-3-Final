import torch
import torch.nn as nn
import os

class PerfectHeuristicNet(nn.Module):
    def __init__(self):
        super(PerfectHeuristicNet, self).__init__()

    def forward(self, ram):
        mi_x = ram[:, 32].float()
        mi_y = ram[:, 34].float()
        su_x = ram[:, 33].float()
        su_y = ram[:, 35].float()

        dist_x = torch.abs(su_x - mi_x)
        dist_y = torch.abs(su_y - mi_y)

        is_pinned_left = (mi_x < 35) & (su_x > mi_x)
        is_pinned_right = (mi_x > 115) & (su_x < mi_x)

        cond_pinned = (is_pinned_left | is_pinned_right) & (dist_x < 35)
        cond_pinned_y_close = dist_y <= 15
        cond_pinned_y_far = ~cond_pinned_y_close

        act_pinned_y_close = torch.where(mi_y < 50, torch.tensor(5), torch.tensor(2))
        act_pinned_y_far = torch.where(is_pinned_left, torch.tensor(3), torch.tensor(4))

        act_pinned = torch.where(
            cond_pinned_y_close, act_pinned_y_close, act_pinned_y_far
        )

        cond_y_align = dist_y > 4 # Stricter Y alignment (was 8)
        act_y_align = torch.where(su_y > mi_y, torch.tensor(5), torch.tensor(2))

        cond_x_far = dist_x > 26 # Closer X alignment (was 28)
        act_x_far = torch.where(su_x > mi_x, torch.tensor(3), torch.tensor(4))

        cond_x_close = dist_x < 22 # Closer minimum distance (was 24)
        act_x_close = torch.where(su_x > mi_x, torch.tensor(4), torch.tensor(3))

        act_punch = torch.tensor(1)

        action = act_punch
        action = torch.where(cond_x_close, act_x_close, action)
        action = torch.where(cond_x_far, act_x_far, action)
        action = torch.where(cond_y_align, act_y_align, action)
        action = torch.where(cond_pinned, act_pinned, action)

        logits = torch.zeros(1, 6)
        logits.scatter_(1, action.unsqueeze(0), 1.0)
        return logits

def export():
    model = PerfectHeuristicNet()
    model.eval()
    dummy_input = torch.zeros(1, 128, dtype=torch.uint8)
    onnx_path = os.path.join(os.path.dirname(__file__), "perfect.onnx")
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"])

if __name__ == "__main__":
    export()

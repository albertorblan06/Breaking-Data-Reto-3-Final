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
        clock = ram[:, 0].float()

        dist_x = torch.abs(su_x - mi_x)
        dist_y = torch.abs(su_y - mi_y)

        go_down = su_y > mi_y
        go_up = su_y < mi_y
        go_right = su_x > mi_x
        go_left = su_x < mi_x

        act_move = torch.tensor(0)

        # 1. Relaxed Y alignment: <= 6
        cond_y_off = dist_y > 6
        act_move = torch.where(cond_y_off & go_up, torch.tensor(2), act_move)
        act_move = torch.where(cond_y_off & go_down, torch.tensor(5), act_move)

        # 2. Relaxed X alignment: Target dist_x between 20 and 32
        cond_y_aligned = dist_y <= 6
        cond_too_far = cond_y_aligned & (dist_x > 32)
        act_move = torch.where(cond_too_far & go_right, torch.tensor(3), act_move)
        act_move = torch.where(cond_too_far & go_left, torch.tensor(4), act_move)

        cond_too_close = cond_y_aligned & (dist_x < 20)
        act_move = torch.where(cond_too_close & go_right, torch.tensor(4), act_move) # BACK AWAY
        act_move = torch.where(cond_too_close & go_left, torch.tensor(3), act_move) # BACK AWAY

        # 3. Sweet spot: 20 <= dist_x <= 32 AND dist_y <= 6
        sweet_spot = cond_y_aligned & (dist_x >= 20) & (dist_x <= 32)

        act_punch = torch.tensor(1)
        act_punch = torch.where(go_right, torch.tensor(11), act_punch) # Punch Right
        act_punch = torch.where(go_left, torch.tensor(12), act_punch)  # Punch Left

        # Punch rhythm: faster rhythm (4 out of every 6 frames)
        punch_now = (clock % 6) < 4

        action = torch.where(sweet_spot & punch_now, act_punch, act_move)

        logits = torch.zeros(1, 18)
        logits.scatter_(1, action.unsqueeze(0), 1.0)
        return logits

def export():
    model = SniperNet()
    model.eval()
    dummy_input = torch.zeros(1, 128, dtype=torch.uint8)
    onnx_path = os.path.join(os.path.dirname(__file__), "aggressive_rhythm.onnx")
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"])

if __name__ == "__main__":
    export()

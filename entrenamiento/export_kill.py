import torch
import torch.nn as nn
import os

class KillNet(nn.Module):
    def __init__(self):
        super(KillNet, self).__init__()

    def forward(self, ram):
        mi_x = ram[:, 32].float()
        mi_y = ram[:, 34].float()
        su_x = ram[:, 33].float()
        su_y = ram[:, 35].float()

        dist_x = torch.abs(su_x - mi_x)
        dist_y = torch.abs(su_y - mi_y)

        go_down = su_y > mi_y
        go_up = su_y < mi_y
        go_right = su_x > mi_x
        go_left = su_x < mi_x

        cond_y = dist_y > 2
        act_y = torch.where(go_down, torch.tensor(5), torch.tensor(2))

        cond_x_far = dist_x > 24
        act_x_far = torch.where(go_right, torch.tensor(3), torch.tensor(4))
        
        cond_x_close = dist_x < 22
        act_x_close = torch.where(go_right, torch.tensor(4), torch.tensor(3)) # move away!

        cond_diag_far = cond_y & cond_x_far
        act_diag_far = torch.tensor(6) # UPRIGHT
        act_diag_far = torch.where(go_up & go_right, torch.tensor(6), act_diag_far)
        act_diag_far = torch.where(go_up & go_left, torch.tensor(7), act_diag_far)
        act_diag_far = torch.where(go_down & go_right, torch.tensor(8), act_diag_far)
        act_diag_far = torch.where(go_down & go_left, torch.tensor(9), act_diag_far)

        cond_diag_close = cond_y & cond_x_close
        act_diag_close = torch.tensor(7) # UPLEFT (backing away diagonally)
        act_diag_close = torch.where(go_up & go_right, torch.tensor(7), act_diag_close) # if they are right, move left
        act_diag_close = torch.where(go_up & go_left, torch.tensor(6), act_diag_close)
        act_diag_close = torch.where(go_down & go_right, torch.tensor(9), act_diag_close)
        act_diag_close = torch.where(go_down & go_left, torch.tensor(8), act_diag_close)

        action = torch.tensor(1) # default PUNCH
        action = torch.where(cond_x_close, act_x_close, action)
        action = torch.where(cond_x_far, act_x_far, action)
        action = torch.where(cond_y, act_y, action)
        action = torch.where(cond_diag_far, act_diag_far, action)
        action = torch.where(cond_diag_close, act_diag_close, action)

        logits = torch.zeros(1, 18)
        logits.scatter_(1, action.unsqueeze(0), 1.0)
        return logits

def export():
    model = KillNet()
    model.eval()
    dummy_input = torch.zeros(1, 128, dtype=torch.uint8)
    onnx_path = os.path.join(os.path.dirname(__file__), "kill.onnx")
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"])

if __name__ == "__main__":
    export()

import torch
import torch.nn as nn
import os

class UnstoppableNet(nn.Module):
    def __init__(self):
        super(UnstoppableNet, self).__init__()

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
        
        # Default: Perfect Punch Range (dist_x == 26)
        act = torch.tensor(1) # PUNCH
        
        # 1. Chasing (dist_x > 26) -> use diagonal fire!
        cond_chase = dist_x > 26
        act_chase = torch.tensor(1)
        act_chase = torch.where(go_up & go_right, torch.tensor(14), act_chase)
        act_chase = torch.where(go_up & go_left, torch.tensor(15), act_chase)
        act_chase = torch.where(go_down & go_right, torch.tensor(16), act_chase)
        act_chase = torch.where(go_down & go_left, torch.tensor(17), act_chase)
        act_chase = torch.where((dist_y < 3) & go_right, torch.tensor(11), act_chase)
        act_chase = torch.where((dist_y < 3) & go_left, torch.tensor(12), act_chase)
        
        # 2. Backing up (dist_x < 26) -> use regular movement (no fire) to dodge!
        cond_dodge = dist_x < 26
        act_dodge = torch.where(go_right, torch.tensor(4), torch.tensor(3)) # Move LEFT if they are RIGHT
        
        # 3. Y-Alignment if dist_x == 26 but Y is off
        cond_align_y = (dist_x == 26) & (dist_y > 2)
        act_align_y = torch.where(go_down, torch.tensor(5), torch.tensor(2))
        
        action = act
        action = torch.where(cond_align_y, act_align_y, action)
        action = torch.where(cond_chase, act_chase, action)
        action = torch.where(cond_dodge, act_dodge, action)

        logits = torch.zeros(1, 18)
        logits.scatter_(1, action.unsqueeze(0), 1.0)
        return logits

def export():
    model = UnstoppableNet()
    model.eval()
    dummy_input = torch.zeros(1, 128, dtype=torch.uint8)
    onnx_path = os.path.join(os.path.dirname(__file__), "unstoppable.onnx")
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"])

if __name__ == "__main__":
    export()

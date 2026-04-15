import torch
import torch.nn as nn
import os

class TrapperNet(nn.Module):
    def __init__(self):
        super(TrapperNet, self).__init__()

    def forward(self, ram):
        mi_x = ram[:, 32].float()
        mi_y = ram[:, 34].float()
        su_x = ram[:, 33].float()
        su_y = ram[:, 35].float()

        dist_x = torch.abs(su_x - mi_x)
        dist_y = torch.abs(su_y - mi_y)

        # 1. Y-Alignment is crucial. If not aligned, align!
        # They are above us -> move UP
        # They are below us -> move DOWN
        cond_y_align = dist_y > 4
        act_y_align = torch.where(su_y < mi_y, torch.tensor(2), torch.tensor(5)) # 2=UP, 5=DOWN

        # 2. X-Trapper Logic
        # If they are on the left side (su_x < 80), trap them from the right: we want mi_x to be slightly > su_x
        # If they are on the right side (su_x >= 80), trap them from the left: we want mi_x to be slightly < su_x
        is_left_side = su_x < 80
        
        target_x_left_side = su_x + 23  # We stand to their right
        target_x_right_side = su_x - 23 # We stand to their left
        
        target_x = torch.where(is_left_side, target_x_left_side, target_x_right_side)
        
        # If our X is far from target X, move towards it
        dist_target_x = mi_x - target_x
        
        # If dist_target_x > 2 (we are too far right of target), move LEFT (4)
        # If dist_target_x < -2 (we are too far left of target), move RIGHT (3)
        cond_x_move_left = dist_target_x > 2
        cond_x_move_right = dist_target_x < -2
        
        act_x = torch.where(cond_x_move_left, torch.tensor(4), torch.tensor(3))
        cond_x_move = cond_x_move_left | cond_x_move_right

        # 3. Punch if in range!
        # If dist_x <= 25 and dist_y <= 6
        cond_punch = (dist_x <= 25) & (dist_y <= 8)
        act_punch = torch.tensor(1)

        # Priority: 
        # 1. If in punch range, PUNCH!
        # 2. Else if Y is misaligned, ALIGN Y!
        # 3. Else if X is misaligned from trap position, ALIGN X!
        # 4. Default: NOOP
        
        action = torch.tensor(0) # Default NOOP
        action = torch.where(cond_x_move, act_x, action)
        action = torch.where(cond_y_align, act_y_align, action)
        action = torch.where(cond_punch, act_punch, action)

        logits = torch.zeros(1, 6)
        logits.scatter_(1, action.unsqueeze(0), 1.0)
        return logits

def export():
    model = TrapperNet()
    model.eval()
    dummy_input = torch.zeros(1, 128, dtype=torch.uint8)
    onnx_path = os.path.join(os.path.dirname(__file__), "trapper.onnx")
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"])

if __name__ == "__main__":
    export()

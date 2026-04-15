import torch
import torch.nn as nn
import os


class RhythmNet(nn.Module):
    def __init__(self):
        super(RhythmNet, self).__init__()

    def forward(self, ram):
        mi_x = ram[:, 32].float()
        mi_y = ram[:, 34].float()
        su_x = ram[:, 33].float()
        su_y = ram[:, 35].float()
        clock = ram[:, 0].float()  # Increments by ~4 each frame

        dist_x = torch.abs(su_x - mi_x)
        dist_y = torch.abs(su_y - mi_y)

        go_down = su_y > mi_y + 1
        go_up = su_y < mi_y - 1
        go_right = su_x > mi_x
        go_left = su_x < mi_x

        # 1. Base Action: Default to 0 (NO-OP)
        act = torch.tensor(0)

        # Rhythm clock: RAM[0] mod 12 (since it increments by ~4, 12 is about 3 frames)
        # clock % 12 < 4 gives true roughly 1 out of 3 frames
        punch_now = (clock % 12) < 4

        # Target Sweet Spot: X=25, Y=0

        # If perfect distance, punch with rhythm!
        perfect_pos = (dist_x >= 23) & (dist_x <= 26) & (dist_y <= 2)
        act_perfect = torch.where(punch_now, torch.tensor(1), torch.tensor(0))

        # If too far in X, close distance
        act_close_x = torch.where(go_right, torch.tensor(3), torch.tensor(4))

        # If too close in X, back away
        act_flee_x = torch.where(go_right, torch.tensor(4), torch.tensor(3))

        # If Y is misaligned, align Y
        act_align_y = torch.where(go_down, torch.tensor(5), torch.tensor(2))

        # Decision Tree Logic
        action = act

        # Y alignment takes priority if Y is off by more than 2
        action = torch.where(dist_y > 2, act_align_y, action)

        # If Y is aligned, check X
        cond_too_far = (dist_y <= 2) & (dist_x > 26)
        action = torch.where(cond_too_far, act_close_x, action)

        cond_too_close = (dist_y <= 2) & (dist_x < 23)
        action = torch.where(cond_too_close, act_flee_x, action)

        # If perfectly aligned in both, PUNCH!
        action = torch.where(perfect_pos, act_perfect, action)

        logits = torch.zeros(1, 18)
        logits.scatter_(1, action.unsqueeze(0), 1.0)
        return logits


def export():
    model = RhythmNet()
    model.eval()
    dummy_input = torch.zeros(1, 128, dtype=torch.uint8)
    onnx_path = os.path.join(os.path.dirname(__file__), "rhythm.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )


if __name__ == "__main__":
    export()

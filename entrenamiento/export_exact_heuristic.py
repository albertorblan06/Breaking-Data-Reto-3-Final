import torch
import torch.nn as nn
import os


class HeuristicNet(nn.Module):
    def __init__(self):
        super(HeuristicNet, self).__init__()

    def forward(self, ram):
        # ram shape: (1, 128)
        # 32: mi_x, 34: mi_y, 33: su_x, 35: su_y

        mi_x = ram[:, 32].float()
        mi_y = ram[:, 34].float()
        su_x = ram[:, 33].float()
        su_y = ram[:, 35].float()

        dist_x = torch.abs(su_x - mi_x)
        dist_y = torch.abs(su_y - mi_y)

        is_pinned_left = (mi_x < 35) & (su_x > mi_x)
        is_pinned_right = (mi_x > 115) & (su_x < mi_x)

        # 0: NOOP, 1: PUNCH, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
        action = torch.zeros(1, dtype=torch.int64)

        # We need to implement the logic with torch.where

        # is_pinned condition
        cond_pinned = (is_pinned_left | is_pinned_right) & (dist_x < 35)
        cond_pinned_y_close = dist_y <= 15
        cond_pinned_y_far = ~cond_pinned_y_close

        act_pinned_y_close = torch.where(mi_y < 50, torch.tensor(5), torch.tensor(2))
        act_pinned_y_far = torch.where(is_pinned_left, torch.tensor(3), torch.tensor(4))

        act_pinned = torch.where(
            cond_pinned_y_close, act_pinned_y_close, act_pinned_y_far
        )

        # else if dist_y > 8
        cond_y_align = dist_y > 8
        act_y_align = torch.where(su_y > mi_y, torch.tensor(5), torch.tensor(2))

        # else if dist_x > 28
        cond_x_far = dist_x > 28
        act_x_far = torch.where(su_x > mi_x, torch.tensor(3), torch.tensor(4))

        # else if dist_x < 24
        cond_x_close = dist_x < 24
        act_x_close = torch.where(su_x > mi_x, torch.tensor(4), torch.tensor(3))

        # else (punch range)
        act_punch = torch.tensor(1)

        # Build the final logic tree from bottom to top
        action = act_punch
        action = torch.where(cond_x_close, act_x_close, action)
        action = torch.where(cond_x_far, act_x_far, action)
        action = torch.where(cond_y_align, act_y_align, action)
        action = torch.where(cond_pinned, act_pinned, action)

        # Return as (1, 6) logits so argmax works, or just return action
        # Let's return logits to match standard
        logits = torch.zeros(1, 6)
        logits.scatter_(1, action.unsqueeze(0), 1.0)

        return logits


def export_heuristic():
    model = HeuristicNet()
    model.eval()

    dummy_input = torch.zeros(1, 128, dtype=torch.uint8)
    onnx_path = os.path.join(os.path.dirname(__file__), "exact_heuristic.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Exact Heuristic ONNX exported to {onnx_path}")


if __name__ == "__main__":
    export_heuristic()

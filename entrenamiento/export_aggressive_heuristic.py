import torch
import torch.nn as nn
import os


class AggressiveHunterNet(nn.Module):
    def __init__(self):
        super(AggressiveHunterNet, self).__init__()

    def forward(self, ram):
        mi_x = ram[:, 32].float()
        mi_y = ram[:, 34].float()
        su_x = ram[:, 33].float()
        su_y = ram[:, 35].float()

        dist_x = torch.abs(su_x - mi_x)
        dist_y = torch.abs(su_y - mi_y)

        is_pinned_left = (mi_x < 35) & (su_x > mi_x)
        is_pinned_right = (mi_x > 115) & (su_x < mi_x)

        # Base logic arrays
        cond_pinned = (is_pinned_left | is_pinned_right) & (dist_x < 35)
        cond_pinned_y_close = dist_y <= 15

        act_pinned_y_close = torch.where(mi_y < 50, torch.tensor(5), torch.tensor(2))
        act_pinned_y_far = torch.where(is_pinned_left, torch.tensor(3), torch.tensor(4))
        act_pinned = torch.where(
            cond_pinned_y_close, act_pinned_y_close, act_pinned_y_far
        )

        # AGGRESSIVE Y-ALIGNMENT (Strictly align Y first if dist_y > 4)
        cond_y_align = dist_y > 4
        act_y_align = torch.where(su_y > mi_y, torch.tensor(5), torch.tensor(2))

        # X DISTANCE CONTROL
        cond_x_far = dist_x > 27
        act_x_far = torch.where(su_x > mi_x, torch.tensor(3), torch.tensor(4))

        cond_x_close = dist_x < 23
        act_x_close = torch.where(su_x > mi_x, torch.tensor(4), torch.tensor(3))

        # PUNCH ZONE (between 23 and 27, and Y aligned)
        act_punch = torch.tensor(1)

        # Build logic bottom-up
        action = act_punch
        action = torch.where(cond_x_close, act_x_close, action)
        action = torch.where(cond_x_far, act_x_far, action)
        action = torch.where(cond_y_align, act_y_align, action)
        action = torch.where(cond_pinned, act_pinned, action)

        logits = torch.zeros(1, 6)
        logits.scatter_(1, action.unsqueeze(0), 1.0)
        return logits


def export_heuristic():
    model = AggressiveHunterNet()
    model.eval()

    dummy_input = torch.zeros(1, 128, dtype=torch.uint8)
    onnx_path = os.path.join(os.path.dirname(__file__), "aggressive_hunter.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Aggressive Heuristic ONNX exported to {onnx_path}")


if __name__ == "__main__":
    export_heuristic()

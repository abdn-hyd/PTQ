import torch
import torch.nn as nn
import math


# setting seed
seed_val = 1247384
torch.manual_seed(seed_val)


# defint a simple linear calculation with PTQ
class PTQ(nn.Module):
    def __init__(
        self,
        d_in: int = 4,
        d_out: int = 4,
        bits: int = 8,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.bits = bits
        self.dtype = getattr(torch, f"int{bits}")

        # define the weights
        self.W = nn.Parameter(torch.randn(d_in, d_out, dtype=torch.float32))
        self.b = nn.Parameter(torch.randn(d_out, 1, dtype=torch.float32))
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.W, mean=0.0, std=math.sqrt(2.0 / self.d_out))
        nn.init.normal_(self.b, mean=0.0, std=math.sqrt(2.0 / self.d_out))

    def UAQ_Quantization(
        self,
        Input: torch.Tensor,
    ):
        q_min, q_max = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
        r_min, r_max = Input.min(), Input.max()
        scale = (r_max - r_min) / (q_max - q_min)

        zero_point = q_max - torch.round(r_max / scale)
        zero_point = torch.clamp(
            zero_point, min=q_min, max=q_max).to(self.dtype)

        tensor_q = torch.round(Input / scale) + zero_point.float()
        tensor_q = torch.clamp(tensor_q, q_min, q_max).to(self.dtype)

        return tensor_q, scale, zero_point

    # Symmetric Uniform Quantization
    def SUQ_Quantization(
        self,
        Input: torch.Tensor,
    ):
        q_min, q_max = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
        scale = torch.abs(Input).max() / q_max

        tensor_q = torch.round(Input / scale)
        tensor_q = torch.clamp(tensor_q, q_min, q_max).to(self.dtype)

        return tensor_q, scale

    # Uniform Affine Quantization
    def UAQ(
        self,
        X: torch.Tensor,
    ):
        # original matmul
        ori_out = self.W @ X + self.b

        # quantization
        W_q, S_w, Z_w = self.UAQ_Quantization(self.W)
        X_q, S_x, Z_x = self.UAQ_Quantization(X)

        # S_w(W_q - Z_w) * S_x(X_q - Z_x) = S_w * S_x * [(W_q - Z_w) * (X_q - Z_x)]
        # which can be seperated with float part and int part
        Scales = S_w * S_x

        # prevent OverflowError
        W_q_shifted = W_q.to(torch.int32) - Z_w.to(torch.int32)
        X_q_shifted = X_q.to(torch.int32) - Z_x.to(torch.int32)
        bias_q = torch.round(self.b / Scales).to(torch.int32)

        Int32_matmul = W_q_shifted @ X_q_shifted + bias_q
        quant_out = Scales * Int32_matmul.float()

        return ori_out, quant_out

    def SUQ(self, X: torch.Tensor):
        # original matmul
        ori_out = self.W @ X + self.b

        # quantization
        W_q, S_w = self.SUQ_Quantization(self.W)
        X_q, S_x = self.SUQ_Quantization(X)

        # S_w(W_q) * S_x(X_q) = S_w * S_x * [W_q * X_q]
        # which can be seperated with float part and int part
        Scales = S_w * S_x
        bias_q = torch.round(self.b / Scales).to(torch.int32)

        Int32_matmul = W_q.to(torch.int32) @ X_q.to(torch.int32) + bias_q
        quant_out = Scales * Int32_matmul.float()

        return ori_out, quant_out

    def forward(
        self,
        X: torch.Tensor,
        method: str = "UAQ",
    ):
        return getattr(self, method)(X)


if __name__ == "__main__":
    method = "SUQ"
    X = torch.randn(4, 1)
    print("Float32 Input:\n", X)
    model = PTQ(4, 4)
    print("Float32 Weights:\n", model.W)

    with torch.no_grad():
        ori_out, quant_out = model(X, method)
        error = torch.abs(ori_out - quant_out)
        print("Error:\n", error)

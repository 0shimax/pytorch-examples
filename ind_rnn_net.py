import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy


class IndRNN(torch.nn.Module):
    def __init__(self, num_input_units, num_units,
                 recurrent_max_abs=25., non_linearity=F.relu):
        super().__init__()

        recurrent_max_abs = min(recurrent_max_abs, 1.0 / numpy.sqrt(num_units))

        self.input_kernel = Parameter(torch.zeros(num_units, num_input_units))
        self.recurrent_kernel = Parameter(
            torch.Tensor(num_units,).uniform_(
                -recurrent_max_abs, recurrent_max_abs))
        self.bias = Parameter(torch.zeros(num_units,))

        self.non_linearity = non_linearity
        self.num_units = num_units
        self.recurrent_max_abs = recurrent_max_abs
        self.reset_state()

    def reset_state(self):
        self.h = Variable(torch.zeros(self.num_units))

    def set_hidden_state(self, state):
        self.h = state

    def clip_recurrent_kernel(self):
        self.recurrent_kernel.data.copy_(
            torch.clamp(self.recurrent_kernel.data,
                        max=self.recurrent_max_abs,
                        min=-self.recurrent_max_abs))

    def forward(self, x):
        self.clip_recurrent_kernel()
        output = self.non_linearity(
            F.linear(x, self.input_kernel, self.bias)
            + F.mul(self.recurrent_kernel, self.h))

        # self.clip_recurrent_kernel()
        # recurrent_update = self.h.mul(
        #     self.recurrent_kernel.expand_as(self.h))
        # gate_inputs += recurrent_update.expand_as(gate_inputs)
        # gate_inputs += self.bias.expand_as(gate_inputs)
        # output = self.non_linearity(gate_inputs)
        return output


class EncDec(torch.nn.Module):

    def __init__(self, in_size, h_unit_size, slot_size, memory_size):
        super().__init__()
        self.encorder = IndRNN(in_size, h_unit_size)
        self.decorder = IndRNN(in_size, h_unit_size)
        self.hidden_to_output = torch.nn.Linear(h_unit_size, in_size)
        self.is_train = True

    def reset_state(self):
        self.encorder.reset_state()
        self.decorder.reset_state()

    def train_step(self, xs, ys):
        for time_idx in reversed(range(1, xs.shape[1])):
            dec_h = self.decorder(xs[:, time_idx])
            out = self.hidden_to_output(dec_h)
            ys.append(out.data)
            self.loss += (xs[:, time_idx-1] - out)**2
        return ys

    def predict(self, xs, ys, out):
        for time_idx in reversed(range(1, xs.shape[1])):
            dec_h = self.decorder(out)
            out = self.hidden_to_output(dec_h)
            ys.append(out.data)
        return ys

    def forward(self, xs):
        n_batch, n_times, dim_obs = xs.shape
        for time_idx in range(n_times):
            x = xs[:, time_idx]
            h = self.encorder(x)

        self.decorder.set_hidden_state(h)
        ys = []
        out = self.hidden_to_output(self.decorder.h)

        ys.append(out)
        self.loss = (xs[:, -1] - out)**2

        if self.is_train:
            self.out = self.train_step(xs, ys)
        else:
            self.out = self.predict(xs, ys, out)

        self.loss /= n_times
        self.loss = self.loss.sum()/n_batch
        return self.loss

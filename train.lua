require 'nn'
require 'nngraph'
require 'autobw'
require 'math'
require 'optim'

local n_input = 1
local n_output = 2
local n_hidden = 25
local batch_size = 15
local seq_length = 5

local function make_rnn_layer(n_input, n_hidden)
    local input = nn.Identity()()
    local prev_state = nn.Identity()()

    local next_state = nn.Sigmoid()(nn.CAddTable()({
        nn.Linear(n_input, n_hidden)(input),
        nn.Linear(n_hidden, n_hidden)(prev_state)
    }))

    output = nn.Identity()(next_state)

    return nn.gModule({input, prev_state}, {output, next_state})
end

local model = {
    -- Add extra layers here (it doesn't help on this problem)
    layers = {
        make_rnn_layer(n_input, n_hidden),
        --make_rnn_layer(n_hidden, n_hidden),
    },
    state = {
        torch.zeros(batch_size, n_hidden),
        --torch.zeros(batch_size, n_hidden),
    },
    output = nn.Linear(n_hidden, n_output),
    criterion = nn.CrossEntropyCriterion(),

    tape = autobw.Tape(),

    forward = function(self, inputs, targets)
        self.tape:start()

        local output
        for t = 1, inputs:size(1) do
            output = inputs[t]
            for l = 1, #self.layers do
                output, self.state[l] = unpack(self.layers[l]:forward({output, self.state[l]}))
            end
        end

        output = self.output:forward(output)
        local loss = self.criterion:forward(output, targets)
        local _, pred = torch.max(output, 2)

        self.tape:stop()

        return loss, torch.eq(pred:int(), targets):float():mean()
    end,

    backward = function(self)
        self.tape:backward()
    end,

    get_parameters = function(self)
        local pack = nn.Sequential()
        for l = 1, #self.layers do
            pack:add(self.layers[l])
        end
        pack:add(self.output)
        return pack:getParameters()
    end
}

local data = torch.linspace(0, 20*math.pi, 1000):sin():view(-1, 1)
local start_idx = torch.Tensor(batch_size):uniform():mul(data:size(1) - seq_length):ceil():long()
local batch = torch.zeros(seq_length, batch_size, 1)

local function next_batch()
    start_idx:add(-1)
    for i = 1, seq_length do
        start_idx:apply(function(x) return x % data:size(1) + 1 end)
        batch:select(1, i):copy(data:index(1, start_idx):view(1, -1, 1))
    end
    return batch:clone()
end

local params, grads = model:get_parameters()
params:uniform(-0.1, 0.1)

local function fopt(x)
    if params ~= x then
        params:copy(x)
    end
    grads:zero()

    local batch = next_batch()
    local inputs = batch:sub(1, batch:size(1)-1)
    local targets = inputs:sum(3):sum(1):view(inputs:size(2))
    for i = 1, targets:size(1) do
      if targets[i] > 0 then targets[i] = 2 else targets[i] = 1 end
    end
    targets = targets:int()

    local loss, acc = model:forward(inputs, targets)
    model:backward()
    print('loss', loss, 'acc', acc)

    return loss, grads
end

for i = 1, 10000 do
    local _, fx = optim.sgd(fopt, params, {})
end


require 'nn'
require 'nngraph'
require 'autobw'
require 'math'
require 'optim'

local n_vocab = 10
local n_emb = 3
local n_output = 4
local n_hidden = 5
local batch_size = 1
local seq_length = 5

local function make_rnn_layer(n_input, n_hidden)
    local input = nn.Identity()()
    local prev_state = nn.Identity()()

    local next_state = nn.Sigmoid()(nn.CAddTable()({
        nn.Linear(n_emb, n_hidden)(input),
        nn.Linear(n_hidden, n_hidden)(prev_state)
    }))

    output = nn.Identity()(next_state)

    return nn.gModule({input, prev_state}, {output, next_state})
end

local model = {
    lookup = nn.LookupTable(n_vocab, n_emb),
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

    forward = function(self, x, y)
        self.tape:start()

        local emb, output
        for t = 1, x:size(1) do
            emb = self.lookup:forward(torch.IntTensor{x[t]})
            output = emb
            for l = 1, #self.layers do
                output, self.state[l] = unpack(self.layers[l]:forward({output, self.state[l]}))
            end
            --print('t', t, 'x', x[t], 'emb', emb, 'h', output)
        end
        output = self.output:forward(output)
        local loss = self.criterion:forward(output, y)
        local _, pred = torch.max(output, 2)

        --print('out', output, 'loss', loss, 'pred', pred)

        self.tape:stop()

        return loss, torch.eq(pred:int(), y):float():mean()
    end,

    backward = function(self)
        self.tape:backward()
    end,

    get_parameters = function(self)
        local pack = nn.Sequential()
        pack:add(self.lookup)
        for l = 1, #self.layers do
            pack:add(self.layers[l])
        end
        pack:add(self.output)
        return pack:getParameters()
    end
}

local data = torch.IntTensor{
  {1, 2, 3},
  {2, 3, 4},
  {3, 4, 1},
}

local labels = torch.IntTensor{4, 1, 2}
local iter = 1

local params, grads = model:get_parameters()
params:uniform(-0.1, 0.1)

local function fopt(x)
    if params ~= x then
        params:copy(x)
    end
    grads:zero()

    local x = data[iter]
    local y = labels[iter]
    iter = ((iter + 1) % data:size(1)) + 1

    local loss, acc = model:forward(x, y)
    model:backward()
    print('loss', loss, 'acc', acc)

    return loss, grads
end

for i = 1, 30000 do
    local _, fx = optim.sgd(fopt, params, {learningRate=1e-2})
end


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
    local x = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    function new_input_sum()
        -- transforms input
        local i2h            = nn.Linear(n_input, n_hidden)(x)
        -- transforms previous timestep's output
        local h2h            = nn.Linear(n_hidden, n_hidden)(prev_h)
        return nn.CAddTable()({i2h, h2h})
    end

    local in_gate          = nn.Sigmoid()(new_input_sum())
    local forget_gate      = nn.Sigmoid()(new_input_sum())
    local out_gate         = nn.Sigmoid()(new_input_sum())
    local in_transform     = nn.Tanh()(new_input_sum())

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    local output           = nn.Identity()(next_h)

    return nn.gModule({x, prev_c, prev_h}, {output, next_c, next_h})
end

local model = {
    lookup = nn.LookupTable(n_vocab, n_emb),
    initial = {
        nn.LookupTable(2, n_hidden),
        --nn.LookupTable(2, n_hidden),
    },
    -- Add extra layers here (it doesn't help on this problem)
    layers = {
        make_rnn_layer(n_emb, n_hidden),
        --make_rnn_layer(n_hidden, n_hidden),
    },
    h_state = {
        torch.zeros(batch_size, n_hidden),
        --torch.zeros(batch_size, n_hidden),
    },
    c_state = {
        torch.zeros(batch_size, n_hidden),
        --torch.zeros(batch_size, n_hidden),
    },
    output = nn.Linear(n_hidden, n_output),
    criterion = nn.CrossEntropyCriterion(),

    tape = autobw.Tape(),

    forward = function(self, x, y)
        self.tape:start()

        -- look up initial states
        for l = 1, #self.layers do
            self.c_state[l] = self.initial[l]:forward(torch.ones(1))
            self.h_state[l] = self.initial[l]:forward(torch.ones(1)+1)
        end

        -- through time
        local emb, output
        for t = 1, x:size(1) do
            -- fetch embeddings
            emb = self.lookup:forward(torch.IntTensor{x[t]})
            output = emb
            for l = 1, #self.layers do
                -- propagate through lstm layers
                output, self.c_state[l], self.h_state[l] = unpack(
                  self.layers[l]:forward({output, self.c_state[l], self.h_state[l]})
                )
            end
            --print('t', t, 'x', x[t], 'emb', emb, 'h', output)
        end
        -- output layer to # classes scores
        output = self.output:forward(output)
        -- compute cross entropy
        local loss = self.criterion:forward(output, y)
        -- predict class
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
            pack:add(self.initial[l])
            pack:add(self.layers[l])
        end
        pack:add(self.output)
        return pack:getParameters()
    end
}

local X = {}
local Y = {}

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

for i = 1, 100 do
    local _, fx = optim.rmsprop(fopt, params, {learningRate=1e-2})
end


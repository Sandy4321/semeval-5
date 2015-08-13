require 'nn'
require 'nngraph'
require 'autobw'
require 'math'
require 'optim'
require 'torchlib'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')

local vocab = torch.load('vocabs.t7')
local n_vocab = vocab.word:size()
local n_emb = 50
local n_output = vocab.rel:size()
local n_hidden = 128
local n_epoch = 10
local batch_size = 1

local senna = torch.load('senna.t7')
local train = torch.load('train.t7')
local iter = 1

torch.manualSeed(123)

local function subsample(data, n)
  local subx = {}
  local suby = {}
  local total = #data.X
  for i = 1, n do
    local subi = math.ceil(math.random() * total)
    table.insert(subx, data.X[subi])
    table.insert(suby, data.Y[subi])
  end
  return {X=subx, Y=suby}
end

--train = subsample(train, 100)

local function make_rnn_layer(n_input, n_hidden)
    local x = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    function new_input_sum(is_forget)
        -- transforms input
        local i2h_linear     = nn.Linear(n_input, n_hidden)
        -- transforms previous timestep's output
        local h2h_linear     = nn.Linear(n_hidden, n_hidden)
        -- initialize
        i2h_linear.weight:uniform(-0.1, 0.1)
        i2h_linear.bias:zero()
        h2h_linear.weight:eye(n_hidden):mul(0.8)
        h2h_linear.bias:zero()
        if is_forget then
          h2h_linear.bias:add(1)
        end

        local i2h            = i2h_linear(x)
        local h2h            = h2h_linear(prev_h)
        return nn.CAddTable()({i2h, h2h})
    end

    local in_gate          = nn.Sigmoid()(new_input_sum())
    local forget_gate      = nn.Sigmoid()(new_input_sum(true))
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

-- initialize
local function glorot(W)
  local n_in = W:size(1)
  local n_out = W:size(2)
  local bound = torch.sqrt(6. / (n_in + n_out))
  return W:uniform(-bound, bound)
end

model.lookup.weight:zero():add(senna)
for l = 1, #model.layers do
  model.initial[l].weight:zero()
end
glorot(model.output.weight)
model.output.bias:zero()

-- define train/test functions
local params, grads = model:get_parameters()

local function fopt(x)
    if params ~= x then
        params:copy(x)
    end
    grads:zero()

    local ind = math.ceil(math.random() * #train.X)
    local x = train.X[ind]
    local y = train.Y[ind]
    iter = (iter % #train.X) + 1

    local loss, acc = model:forward(x, y)
    model:backward()
    --print('loss', loss, 'acc', acc)

    return loss, grads
end

local function fpredict(data)
  local total_loss = 0
  local total_acc = 0
  for i = 1, #data.X do
    local x = data.X[i]
    local y = data.Y[i]
    local loss, acc = model:forward(x, y)
    total_loss = total_loss + loss
    total_acc = total_acc + acc
  end
  return total_loss / #data.X, total_acc / #data.X
end

for epoch = 1, n_epoch do
    -- train epoch
    optim.rmsprop(fopt, params, {learningRate=1e-2})
    xlua.progress(iter, #train.X)
    while iter ~= 1 do
      _, fx = optim.rmsprop(fopt, params, {learningRate=1e-2})
      xlua.progress(iter, #train.X)
    end
    -- evaluate
    local loss, acc = fpredict(train)
    print('epoch', epoch, 'loss', loss, 'acc', acc)
end


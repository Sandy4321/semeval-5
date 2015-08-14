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
local cuda = false
local gpu = 0

local senna = torch.load('senna.t7')
local train = torch.load('train.t7')
local iter = 1

torch.manualSeed(123)

if cuda then
  require 'cunn'
  require 'cutorch'
  cutorch.setDevice(0)
  cutorch.manualSeed(123)
else
  local fakecuda = require 'fakecuda'
  fakecuda.init(true)
end


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

local function glorot(W, n_in, n_out)
  n_in = n_in or W:size(1)
  n_out = n_out or W:size(2)
  local bound = torch.sqrt(6. / (n_in + n_out))
  return W:uniform(-bound, bound)
end

local function make_rnn_layer(n_input, n_hidden)
    local x = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    local x = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    local i2h_linear = nn.Linear(n_input, 4 * n_hidden)
    local h2h_linear = nn.Linear(n_hidden, 4 * n_hidden)
    glorot(i2h_linear.weight, n_input, n_hidden)
    glorot(h2h_linear.weight, n_hidden, n_hidden)
    local i2h = i2h_linear(x)
    local h2h = h2h_linear(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local sigmoid_chunk = nn.Narrow(2, 1, 3 * n_hidden)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, n_hidden)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, n_hidden + 1, n_hidden)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, 2 * n_hidden + 1, n_hidden)(sigmoid_chunk)

    local in_transform = nn.Narrow(2, 3 * n_hidden + 1, n_hidden)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)

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


model.lookup.weight:zero():add(senna)
for l = 1, #model.layers do
  model.initial[l].weight:zero()
end
glorot(model.output.weight)
model.output.bias:zero()

-- define train/test functions
local params, grads = model:get_parameters()
params:cuda()

local function fopt(x)
    if params ~= x then
        params:copy(x)
    end
    grads:zero()

    local ind = math.ceil(math.random() * #train.X)
    local x = train.X[ind]
    local y = train.Y[ind]

    local loss, acc = model:forward(x, y)
    model:backward()
    --print('loss', loss, 'acc', acc)

    return loss, grads
end

local function fpredict(data)
  local total_loss = 0
  local total_acc = 0
  for i = 1, #data.X do
    local x = data.X[i]:cuda()
    local y = data.Y[i]
    local loss, acc = model:forward(x, y)
    total_loss = total_loss + loss
    total_acc = total_acc + acc
  end
  return total_loss / #data.X, total_acc / #data.X
end

for epoch = 1, n_epoch do
    -- train epoch
    for iter = 1, #train.X do
      _, fx = optim.rmsprop(fopt, params, {learningRate=1e-2})
      xlua.progress(iter, #train.X)
    end
    xlua.progress(#train.X, #train.X)
    -- evaluate
    local loss, acc = fpredict(train)
    print('epoch', epoch, 'loss', loss, 'acc', acc)
    torch.save('params.' .. epoch .. '.t7', params)
end


require 'torch'
require 'torchlib'
require 'nn'
require 'nngraph'
require 'optim'
local LSTM = require 'LSTM'             -- LSTM timestep and utilities
require 'Embedding'                     -- class name is Embedding (not namespaced)
local model_utils = require 'model_utils'
nngraph.setDebug(true)


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple character-level LSTM language model')
cmd:text()
cmd:text('Options')
cmd:option('-vocabfile','vocabs.t7','filename of the string->int tables')
cmd:option('-trainfile','train.t7','filename of the serialized torch ByteTensor to load')
cmd:option('-testfile','test.t7','filename of the serialized torch ByteTensor to load')
cmd:option('-batch_size',1,'number of sequences to train on in parallel')
cmd:option('-seq_length',16,'number of timesteps to unroll to')
cmd:option('-rnn_size',256,'size of LSTM internal state')
cmd:option('-max_epochs',1,'number of full passes through the training data')
cmd:option('-savefile','model_autosave','filename to autosave the model (protos) to, appended with the,param,string.t7')
cmd:option('-save_every',100,'save every 100 steps, overwriting the existing file')
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)

-- preparation stuff:
torch.manualSeed(opt.seed)
opt.savefile = cmd:string(opt.savefile, opt,
    {save_every=true, print_every=true, savefile=true, vocabfile=true, datafile=true})
    .. '.t7'

local vocabs = torch.load(opt.vocabfile)
local vocab_size = vocabs.word:size()

-- define model prototypes for ONE timestep, then clone them
--
local protos = {}
protos.embed = Embedding(vocabs.word:size(), opt.rnn_size)
-- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}
protos.lstm = LSTM.lstm(opt)
protos.softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, vocab_size)):add(nn.LogSoftMax())
protos.criterion = nn.ClassNLLCriterion()

-- put the above things into one flattened parameters tensor
local params, grad_params = model_utils.combine_all_parameters(protos.embed, protos.lstm, protos.softmax)
params:uniform(-0.08, 0.08)


local Loader = torch.class('Loader')
function Loader:__init(fname)
  self.data = torch.load(fname)
  self.i = 1
  self.nbatches = #self.data.X
end

function Loader:next_batch()
  local x = self.data.X[self.i]
  x = x:reshape(1, x:size(1))
  local y = self.data.Y[self.i]
  self.i = self.i + 1
  return x, y
end

local Dtrain = Loader.new(opt.trainfile)
local Dtest = Loader.new(opt.testfile)

-- do fwd/bwd and return loss, grad_params
function feval(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = Dtrain:next_batch()
    local unroll = x:size(1)

    ------------------ unroll --------------------------
    -- make a bunch of clones, AFTER flattening, as that reallocates memory
    local clones = {}
    for name,proto in pairs(protos) do
        print('cloning '..name)
        clones[name] = model_utils.clone_many_times(proto, unroll, not proto.parameters)
    end

    -- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
    local initstate_c = torch.zeros(opt.batch_size, opt.rnn_size)
    local initstate_h = initstate_c:clone()

    -- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
    local dfinalstate_c = initstate_c:clone()
    local dfinalstate_h = initstate_c:clone()

    ------------------- forward pass -------------------
    local embeddings = {}            -- input embeddings
    local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=initstate_h} -- output values of LSTM
    local predictions = {}           -- softmax outputs
    local loss = 0

    for t=1, unroll do
        embeddings[t] = clones.embed[t]:forward(x[t])

        -- we're feeding the *correct* things in here, alternatively
        -- we could sample from the previous timestep and embed that, but that's
        -- more commonly done for LSTM encoder-decoder models
        lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})
    end
    predictions[unroll] = clones.softmax[unroll]:forward(lstm_h[unroll])
    loss = clones.criterion[unroll]:forward(predictions[unroll], y)

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dembeddings = {}                              -- d loss / d input embeddings
    local dlstm_c = {[unroll]=dfinalstate_c}    -- internal cell states of LSTM
    local dlstm_h = {}                                  -- output values of LSTM
    for t=unroll,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        -- Two cases for dloss/dh_t:
        --   1. h_T is only used once, sent to the softmax (but not to the next LSTM timestep).
        --   2. h_t is used twice, for the softmax and for the next step. To obey the
        --      multivariate chain rule, we add them.
        if t == opt.seq_length then
            assert(dlstm_h[t] == nil)
            dlstm_h[t] = clones.softmax[t]:backward(lstm_h[t], doutput_t)
        else
            dlstm_h[t]:add(clones.softmax[t]:backward(lstm_h[t], doutput_t))
        end

        -- backprop through LSTM timestep
        dembeddings[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t]:backward(
            {embeddings[t], lstm_c[t-1], lstm_h[t-1]},
            {dlstm_c[t], dlstm_h[t]}
        ))

        -- backprop through embeddings
        clones.embed[t]:backward(x[{{}, t}], dembeddings[t])
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    initstate_c:copy(lstm_c[#lstm_c])
    initstate_h:copy(lstm_h[#lstm_h])

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end

-- optimization stuff
local losses = {}
local optim_state = {learningRate = 1e-1}
local iterations = opt.max_epochs * Dtrain.nbatches
for i = 1, iterations do
    local _, loss = optim.adagrad(feval, params, optim_state)
    losses[#losses + 1] = loss[1]

    if i % opt.save_every == 0 then
        torch.save(opt.savefile, protos)
    end
    if i % opt.print_every == 0 then
        print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, loss[1], loss[1] / opt.seq_length, grad_params:norm()))
    end
end


require 'torch'
require 'torchlib'

function tokenize(example)
  -- add space to make the markup also tokens
  example = string.gsub(example, '<e1>', 'E1START ')
  example = string.gsub(example, '</e1>', ' E1END')
  example = string.gsub(example, '<e2>', 'E2START ')
  example = string.gsub(example, '</e2>', ' E2END')
  local normalized = {}
  for i = 1, string.len(example) do
    local char = string.sub(example, i, i)
    if string.match(char, "%p") then
      table.insert(normalized, ' ' .. char .. ' ')
    else
      table.insert(normalized, char)
    end
  end
  normalized = string.lower(table.concat(normalized))
  local tokens = ArrayList.new()
  for token in string.gmatch(normalized, "%S+") do
    tokens:add(token)
  end
  tokens = tokens:sublist(2) -- first number is example number
  return tokens
end

-- Read files line by line
function readFile(fname, vocabs, add)
  add = add or false
  local function processCache(cache)
    -- process a collection of lines corresponding to one example
    local tokens = tokenize(cache[1]) -- first line is example
    local relation = string.gsub(cache[2], "%s+", '') -- second line is label
    local indices = vocabs.word:indicesOf(tokens:toTable(), add)
    local relationIndex = vocabs.rel:indexOf(relation, true)
    return torch.IntTensor(indices), relationIndex
  end
  local fh, err = io.open(fname)
  if err then
    error('Error: cannot open file ' .. fname)
  end
  local cache = {}
  local X = {}
  local Y = {}
  while true do
    local line = fh:read()
    if line == nil then break end
    if line == '' then
      local words, rel = processCache(cache)
      table.insert(X, words)
      table.insert(Y, rel)
      cache = {}
    else
      table.insert(cache, line)
    end
  end
  return X, Y
end

-- read training file
local vocabs = {word=Vocab.new('UNKNOWN'), rel=Vocab.new()}
print('numericalizing training data')
local Xtrain, Ytrain = readFile('dataset/train.txt', vocabs, true)
print('numericalizing dev data')
local Xdev, Ydev = readFile('dataset/dev.txt', vocabs, false)
print('numericalizing test data')
local Xtest, Ytest = readFile('dataset/test.txt', vocabs, false)

print('train ' .. #Xtrain)
print('dev ' .. #Xdev)
print('test ' .. #Xtest)
print('words ' .. vocabs.word:size())
print('rel ' .. vocabs.rel:size())

torch.save('vocabs.t7', vocabs)
torch.save('train.t7', {X=Xtrain, Y=Ytrain})
torch.save('dev.t7', {X=Xdev, Y=Ydev})
torch.save('test.t7', {X=Xtest, Y=Ytest})


-----------------------------------------------------------------------
--[[ RemoveMatch ]]--
-- A Preprocess that transforms a container by acting upon
-- the first cluster of modules matched by the given conf.
-----------------------------------------------------------------------
local RemoveMatch = torch.class("dp.RemoveMatch", "dp.TransformMatch")
RemoveMatch.isRemoveMatch = true

function RemoveMatch:__init(config)
   config = config or {}
   assert(not config[1], "Constructor requires key-value arguments")
   local args
   args, self._contains, self._start_at, self._match_count, self._global, self._begins_with, self._ends_with
      = xlua.unpack(
      {config},
      'RemoveMatch', nil,
      {arg='contains', type='table', default={},
        help='Match group of Modules that contain these types of Modules.'},
      {arg='start_at', type='number', default=1,
        help='Match group not before this Module index.'},
      {arg='match_count', type='table', default=1,
        help='Match group of at most this many Modules.'},
      {arg='global', type='boolean', default=false,
        help='Match all groups.'},
      {arg='begins_with', type='table', default={},
        help='Match group of Modules that begins with this type of Module.'},
      {arg='ends_with', type='table', default={},
        help='Match group of Modules that ends with this type of Module.'}
   )
end

function RemoveMatch:apply(module)
  assert(#self._contains > 0, "'contains' argument is required")

  local J={}
  for i,v in ipairs(module.modules) do
    if _.contains(self._contains, torch.type(v)) then table.insert(J,i) end
  end

  if not self._global then
    istart,iend = self:_match(module, self._start_at, self._begins_with, self._contains, self._ends_with, self._match_count)
    module = RemoveMatch:_operate(module,istart,iend)
  else
    for i,v in ipairs(J) do
      istart,iend = self:_match(module, self._start_at, self._begins_with, self._contains, self._ends_with, self._match_count)
      module = RemoveMatch:_operate(module,istart,iend)
    end
  end
  return module
end

function RemoveMatch:_operate(module, istart, iend)
  for i=iend,istart,-1 do if istart > 0 then module:remove(i) end end
  return module
end

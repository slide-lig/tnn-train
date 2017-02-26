-----------------------------------------------------------------------
--[[ InsertAtMatch ]]--
-- A Preprocess that transforms a container by acting upon
-- the first cluster of modules matched by the given conf.
-----------------------------------------------------------------------
local InsertAtMatch = torch.class("dp.InsertAtMatch", "dp.TransformMatch")
InsertAtMatch.isInsertAtMatch = true

function InsertAtMatch:__init(config)
   config = config or {}
   assert(not config[1], "Constructor requires key-value arguments")
   local args
   args, self._contains, self._start_at, self._match_count, self._global, self._begins_with, self._ends_with, self.insert_after, self._add_module
      = xlua.unpack(
      {config},
      'InsertAtMatch', nil,
      {arg='contains', type='table', default={},
        help='Match group of Modules that contain these types of Modules.'},
      {arg='start_at', type='number', default=1,
        help='Match group not before this Module index.'},
      {arg='match_count', type='number', default=1,
        help='Match group of at most this many Modules.'},
      {arg='global', type='boolean', default=false,
        help='Match all groups.'},
      {arg='begins_with', type='table', default={},
        help='Match group of Modules that begins with this type of Module.'},
      {arg='ends_with', type='table', default={},
        help='Match group of Modules that ends with this type of Module.'},
      {arg='insert_after', type='boolean', default=false,
        help='Insert the module after the matching group.'},
      {arg='add_module', type='table',
        help='Insert this Module on each index in the matching group of Modules'}
   )
end

function InsertAtMatch:apply(module)
  assert(#self._contains > 0, "'contains' argument is required")

  local J={}
  for i,v in ipairs(module.modules) do
    if _.contains(self._contains, torch.type(v)) then table.insert(J,i) end
  end

  if not self._global then
    istart,iend = self:_match(module, self._start_at, self._begins_with, self._contains, self._ends_with, self._match_count)
    module = InsertAtMatch:_operate(module,istart,iend,self._add_module,self.insert_after)
  else
    for i,v in ipairs(J) do
      istart,iend = self:_match(module, self._start_at, self._begins_with, self._contains, self._ends_with, self._match_count)
      module = InsertAtMatch:_operate(module,istart,iend,self._add_module,self.insert_after)
      self._start_at = math.max(iend+1,self._start_at)
    end
  end
  return module
end

function InsertAtMatch:_operate(module, istart, iend, add_module, insert_after)
  if insert_after and iend>0 and iend<=#module.modules then
    module:insert(add_module,iend+1)
  elseif istart>0 and istart<=#module.modules then
    module:insert(add_module,istart)
  end
  return module
end

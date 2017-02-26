------------------------------------------------------------------------
--[[ TransformMatch ]]--
--Abstract class.
------------------------------------------------------------------------
local TransformMatch = torch.class("dp.TransformMatch")
TransformMatch.isTransformMatch = true

-- module: The Module to act upon. An instance of nn.Module.
function TransformMatch:apply(module)
   error("TransformMatch subclass does not implement an apply method.")
end

function TransformMatch:_match(module, start_at, begins_with, contains, ends_with, match_count)

  istart = -1
  iend = -1
  finished = false

  local I = {}
  local J = {}
  local K = {}

  for i,v in ipairs(module.modules) do
   if _.contains(begins_with, torch.type(v)) then table.insert(I,i) end
   if _.contains(contains, torch.type(v)) then table.insert(J,i) end
   if _.contains(ends_with, torch.type(v)) then table.insert(K,i) end
  end

  for i,v in ipairs(J) do
    if ((#I==0 and v>=start_at) or (_.contains(I,v-1) and v-1>=start_at)) then
      istart = (#I==0 or match_count <= 1) and v or v-1
      iend = v
      while iend <= J[#J] do
        if _.contains(J,iend+1) then
          iend=iend+1
        elseif
          ((match_count >= iend-istart+1) and #K==0) or
          ((match_count >= iend-istart) and _.contains(K,iend+1)) then
            iend = _.contains(K,iend+1) and iend+1 or iend
            finished = true
            break
        end
      end

      if finished then break end

    end
  end

  iend = math.min(iend,istart+match_count-1)
  --print("--Acting upon indexes ["..istart.."]->["..iend.."]")
  return istart,iend

end


List = {}

function List.new (lim)

  return {first = 0, last = -1, limit = lim or 0}
end

function List.count(list)
  return list.last - list.first + 1
end

function List.pushfirst (list, value)
  if list.limit <= List.count(list) then return end
  local first = list.first - 1
  list.first = first
  list[first] = value
end

function List.pushlast (list, value)
  if list.limit <= List.count(list) then return end
  local last = list.last + 1
  list.last = last
  list[last] = value
end

function List.popfirst (list)
  local first = list.first
  if first > list.last then error("list is empty") end
  local value = list[first]
  list[first] = nil        -- to allow garbage collection
  list.first = first + 1
  return value
end

function List.poplast (list)
  local last = list.last
  if list.first > last then error("list is empty") end
  local value = list[last]
  list[last] = nil         -- to allow garbage collection
  list.last = last - 1
  return value
end

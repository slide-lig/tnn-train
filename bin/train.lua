package.path = '../lib/?;../lib/?.lua;'..package.path

require 'utils.DataHandler'
require 'utils.ConfHandler'
require 'Runtime.Coredp'

if #arg <= 0 then
  print("<main> missing arguments")
  os.exit()
end

-- compile configurations (from parts or composite confs)
local configs = ConfHandler.GetFullConfs(arg)
-- run Core over each
tablex.map(Coredp.run,configs)

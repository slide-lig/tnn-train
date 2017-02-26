gnuplotx={}
function gnuplotx.hist_nofx(tensor,bins,min,max)
   local h = gnuplot.histc(tensor,bins,min,max)
   local x_axis = torch.Tensor(#h)
   for i = 1,#h do
      x_axis[i] = h[i].val
   end
   --gnuplot.bar(x_axis, h.raw)
   return h
end

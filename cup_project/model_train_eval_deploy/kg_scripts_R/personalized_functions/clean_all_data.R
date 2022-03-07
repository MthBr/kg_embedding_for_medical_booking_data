#########################################################
### A) Clean all and Load Data
#########################################################
closeAllConnections()
rm(list=ls()) #clear workspace

#clear all Graphics devices
if(!is.null(dev.list())) dev.off()
graphics.off()
#while (rgl.cur()) rgl.close()
#while (dev.cur()>1) dev.off()
#ERROR: BUG
# https://github.com/rstudio/rstudio/issues/3117
#rm(list=ls())
#x11();dev.off()
dev.new() #workaround of bug

closeAllConnections()
rm(list = ls(all.names = TRUE)) #will clear all objects includes hidden objects.
gc() #free up memrory and report the memory usage.


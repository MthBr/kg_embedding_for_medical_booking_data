#SET WORKING DIRECTORY BY COMMAND########################
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

#########################################################
### A) Clean all data
#########################################################
functions_path <- "personalized_functions"
source(paste(functions_path,"clean_all_data.R", sep="/"))

#########################################################
### B) Set local paths
#########################################################
functions_path <- "personalized_functions"
images_path <- "../../../../Rcodes/images"  #image path in local for server!
csv_path <- "../../../../Rcodes/csv"
rds_path <- "../../../../Rcodes/rds"

#########################################################
### C)  Create directories
#########################################################
dir.create(images_path,recursive = TRUE)
dir.create(csv_path,recursive = TRUE)
dir.create(rds_path,recursive = TRUE)




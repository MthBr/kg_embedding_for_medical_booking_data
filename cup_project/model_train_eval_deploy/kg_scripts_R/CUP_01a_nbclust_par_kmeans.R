setwd(dirname(rstudioapi::getSourceEditorContext()$path))
source("starting_script.R")

#############################################################################
### Load data
#############################################################################
source("CUP_00_load_data.R")
# head(df_real_valued)

# PROJECT_NAME <- "italy_crime_index"
# max_nClusters <- 11
# df_real_valued <- read.csv("crime_index_scaled.csv", row.names = 1)
# dataset_matrix <- as.matrix(df_real_valued[4:22])

images_path <- paste(images_path, PROJECT_NAME, sep = "/")
csv_path <- paste(csv_path, PROJECT_NAME, sep = "/")
rds_path <- paste(rds_path, PROJECT_NAME, sep = "/")

#############################################################################
### Add libraries
#############################################################################
library(NbClust)
library(lubridate)
library(parallel)
library(MASS)
library(foreach)
library(doParallel)
library(dplyr)
library(grDevices) #colorramppalette
library(RColorBrewer)
library(iterators)

#############################################################################
### Parallel settings
#############################################################################
numCores <- detectCores()

##### For Windows
#cluster <- makeCluster(3)

##### For Linux
cluster <- 14

registerDoParallel(cluster)
#############################################################################
min_nClusters <- 2
# max_nClusters <- 89 #sqrt(nrow(df_real_valued)) max number of clusters up to study!

# indices = c("mcclain", "cindex", "silhouette", "dunn")
indices = c("kl", "ch", "hartigan", "cindex", "db", "silhouette", "ratkowsky", "ball", "ptbiserial", "mcclain", "dunn", "sdindex", "dindex", "sdbw")
# The TSS matrix is indefinite. There must be too many missing values. The index cannot be calculated. , "ccc", "scott", "marriot", "trcovw", "tracew", "friedman", "rubin"
# Errore: cannot allocate vector of size 8694973.8 Gb , "gap"
# Troppo lento , "gamma", "gplus", "tau"
# Error in cutree(hc, k = best.nc) : oggetto "best.nc" non trovato , "duda", "pseudot2", "beale", "hubert", "frey"
method_nbclust = c("ward.D2", "single", "complete", "average","kmeans") #centroid

x_axes <- min_nClusters:max_nClusters #create x-axes

#############################################################################
# Start Parallelization
#############################################################################
start_time <- Sys.time()#count time

# dataset_matrix <- as.matrix(df_real_valued[1:DIM])
max_nClusters <- max(min(max_nClusters, nrow(dataset_matrix)-2),min_nClusters+2)  
# "number of cluster centres must lie between 1 and nrow(x)"
#"The difference between the minimum and the maximum number of clusters must be at least equal to 2"

df_values <- setNames(data.frame(matrix(ncol = 4, nrow = 0)), c("distance","linkage","index_nbclust", "values_index" )) #INIZIALIZE

df_values = foreach (nbindices = iter(indices), .combine=rbind, .packages = c("foreach","ggplot2", "dplyr")) %:%
  foreach (nbNamMeth = iter(method_nbclust), .combine=rbind, .packages = c("foreach","ggplot2","dplyr")) %dopar% {
    
    strNamMeth <- "euclidean"
    df_4par_values <- setNames(data.frame(matrix(ncol = 4, nrow = 0)), c("distance","linkage","index_nbclust", "values_index" )) #INIZIALIZE
    res <- NULL
    
    res <- NbClust::NbClust(data = dataset_matrix, min.nc = min_nClusters, max.nc = max_nClusters, method = nbNamMeth, index = nbindices)
    
    df_4par_values <- data.frame("distance"=strNamMeth,"linkage"=nbNamMeth, "index_nbclust"= nbindices)
    df_4par_values$values_index <- paste("c(",paste(res$All.index,collapse = ","),")",collapse = "",sep = "")
    
    path_dir <- paste(csv_path, "nbclust", sep = "/" )
    dir.create(path_dir,recursive = TRUE)
    file_name <- paste( "df_nb_clust_values",paste(strNamMeth,nbNamMeth,nbindices,sep = "_"),"_n_start_" , min_nClusters, "_end_", max_nClusters, sep = "")
    path_file<- paste( path_dir, "/", file_name, ".csv", sep = "")
    write.table(df_4par_values, file=path_file, quote=T, sep=";", dec=".", na="", row.names=F, col.names=T, eol = "\n")
    
    
    df_4par_values
  }

end_time <- Sys.time()#end count time
print(paste("Tempo totale",as.period(round(as.duration(end_time-start_time)))))#how much

path_dir <- paste(csv_path, "nbclust", sep = "/" )
dir.create(path_dir,recursive = TRUE)
file_name <- paste( "df_nb_clust_values","_n_start_" , min_nClusters, "_end_", max_nClusters, sep = "")
path_file<- paste( path_dir, "/", file_name, ".csv", sep = "")
write.table(df_values, file=path_file, quote=T, sep=";", dec=".", na="", row.names=F, col.names=T, eol = "\n")

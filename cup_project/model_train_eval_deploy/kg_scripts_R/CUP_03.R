setwd(dirname(rstudioapi::getSourceEditorContext()$path))
source("starting_script.R")
start_time <- Sys.time()#count time
#############################################################################
### Load data
#############################################################################
source("CUP_00_load_data.R")
df_real_valued <- py_modul$get_all_concept_embeddings()

images_path <- paste(images_path, PROJECT_NAME, sep = "/")
csv_path <- paste(csv_path, PROJECT_NAME, sep = "/")
rds_path <- paste(rds_path, PROJECT_NAME, sep = "/")



kmedoids <- list()
basename.matches <- list.files( path = rds_path, recursive = TRUE)
for (elem in basename.matches) {
  filename <- tail(unlist(strsplit(elem,split="/")),n=1)          #get name of all files in directory
  basename <- unlist(strsplit(filename,"[.]"))[1]                 # just remove fomr filename the extension
  splitted_name <- tail(unlist(strsplit(basename,split="_")),n=1) # slect specific pieces of the base name
  print(splitted_name)
  kmedoids[[paste(splitted_name, collapse = "_")]] <- readRDS(paste(rds_path, elem, sep = "/"))
}

#############################################################################
### C)  Create directories
#############################################################################
dir.create(images_path,recursive = TRUE)
dir.create(csv_path,recursive = TRUE)
dir.create(rds_path,recursive = TRUE)

#############################################################################
### Add libraries
#############################################################################
library(lubridate)  # for nice time
library(PCAmixdata)
library(factoextra)
library(dplyr)
source(paste(functions_path,"ggbiplot.R", sep="/"))

#############################################################################
### Parameters to fix
#############################################################################
print(names(df_real_valued))
K_list <- list('person' = 8, 
               'company' = 2,
               'call' = 15,
               'contract' = 2)


#############################################################################
#Principal Component Analysis 
#############################################################################
for (cluster_name in names(df_real_valued)) {
  dataset_matrix <- t(as.matrix(as.data.frame(df_real_valued[cluster_name])))
  print(cluster_name)
  if (nrow(dataset_matrix) < 2)
    next
  
  pca_norm <- prcomp(dataset_matrix,   center = T,  scale. = T)  #calcoliamo le componenti principali
  pca <- prcomp(dataset_matrix,   center = F,  scale. = F)  #calcoliamo le componenti principali
  # applicando fviz_contrib ( che troviamo nel package factoextra)
  #calcoliamo i contributi in termini di correlazione delle componenti principali rispetto alle features di partenza.
  #Verranno considerate “predittive”solo quelle fetures con livello di correlazione > 1/n (con n numero di features)
  plot(pca, type = "l")
  plot(pca_norm, type = "l")
  fviz_contrib(pca, choice = "var", axes = 1, top = 15) 
  fviz_contrib(pca_norm, choice = "var", axes = 1, top = 19) 
  
  print("asdsadsadsadas")
  img <- NULL
  img <- ggbiplot(pca_norm, obs.scale = 1, var.scale = 1,
                  groups = as.factor(kmedoids[[cluster_name]]$clustering), ellipse = FALSE, circle = FALSE) +
    # scale_colour_manual(name = '', values = cols[[month]]) +
    scale_color_brewer(palette = "Set1") +
    theme(legend.direction = 'horizontal', legend.position = 'top')
  
  path_name <- paste(images_path, "PCA", sep = "/")
  dir.create(path_name,recursive = TRUE)
  
  image_name <-paste("PCA_no_ellipse", cluster_name ,"low_res", sep = "_" )
  ggsave(paste(image_name,'.jpeg', sep = ""), plot = img, path = path_name, scale = 1, units = "cm", dpi = "retina")
  
  image_name <-paste("PCA_no_ellipse", cluster_name,"high_res", sep = "_" )
  ggsave(paste(image_name,'.jpeg', sep = ""), plot = img, path = path_name, scale = 4, units = "cm", dpi = 700, limitsize = FALSE)
}

#############################################################################
end_time <- Sys.time()#end count time
print(paste("Tempo totale",as.period(round(as.duration(end_time-start_time)))))#how much
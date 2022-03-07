setwd(dirname(rstudioapi::getSourceEditorContext()$path))
source("starting_script.R")
start_time <- Sys.time()#count time
#############################################################################
### Load data
#############################################################################
source("CUP_00_load_data.R")

images_path <- paste(images_path, PROJECT_NAME, sep = "/")
csv_path <- paste(csv_path, PROJECT_NAME, sep = "/")
rds_path <- paste(rds_path, PROJECT_NAME, sep = "/")

#########################################################
### C)  Create directories
#########################################################
dir.create(images_path,recursive = TRUE)
dir.create(csv_path,recursive = TRUE)
dir.create(rds_path,recursive = TRUE)

#############################################################################
### Add libraries
#############################################################################
library(lubridate)  # for nice time
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering visualization
library(dendextend) # for comparing two dendrograms
library(ggrepel)    # for overlapping text
library(RColorBrewer) # for colours

#############################################################################
### Parameters to fix
#############################################################################
K <- 13
selected_K <- 8

#Sub fast testing part:
set.seed(0)
# nSample <- 100 #number of sample # only for small subtests
# df_MANN_full <- df_MANN
# df_MANN <- df_MANN[sample(nrow(df_MANN), nSample), ]

#############################################################################
### B2) Parallel settings
#############################################################################
# numCores <- detectCores()
#
# For Windows
# cluster <- makeCluster(3)
# For Linux
# cluster <- 8
#
# registerDoParallel(cluster)

#############################################################################
# Hierarchical Clustering Phase 1
#############################################################################
start_time_local <- Sys.time()#count time

# Dissimilarity matrix
d <- dist(df_real_valued[1:DIM], method = "euclidean")

# Hierarchical clustering using Ward Linkage
hc <- hclust(d, method = "ward.D2")

# Plot the obtained dendrogram
plot(hc, cex = 0.6, hang = -1)

# Cut tree into 4 groups
sub_grp <- cutree(hc, k = K)

# Number of members in each cluster
table(sub_grp)

# add clusters labels
df_real_valued %>%
  mutate(cluster = sub_grp) %>%
  head

# KMeans
clust <- kmeans(df_real_valued[1:DIM], K, nstart = 50)

# df_real_valued["clusters"] = sub_grp
df_real_valued["clusters"] = clust["cluster"]

# Save to CSV
write.csv(df_real_valued["clusters"], paste(csv_path, paste0(PROJECT_NAME, '_clusters.csv'), sep = "/"), row.names = TRUE)

plot(hc, cex = 0.6)
rect.hclust(hc, k = K, border = 2:5)

#last n cols
df_real_valued[(DIM+1):ncol(df_real_valued)]

#Create a custom color scale
myColors <- brewer.pal(length(table(df_real_valued["entity"])), "Set2")
names(myColors) <- levels(as.factor(df_real_valued$entity))
fillScale <- scale_fill_manual(name = "entity",values = myColors)

p <- ggplot(df_real_valued[(DIM+1):ncol(df_real_valued)], aes(x=factor(clusters), fill=entity))+
  geom_bar(stat="count")+
  fillScale+
  labs(x = "K", y= "")+
  theme_minimal()

image_name <-paste("distribution_entities_Kmax", k, sep = "_" )
ggsave(paste(image_name,'.png', sep = ""), plot = p, path = images_path, dpi = 300, width = 20, units = "cm") # , width = 20, units = "cm"

# df_real_valued[(DIM+1):ncol(df_real_valued)] %>% 
#   count(clusters, entity) -> df_plot
# 
# p <- ggplot(df_plot, aes(x=reorder(factor(clusters), -n), y=n, fill=entity))+
#   geom_bar(stat="identity")
# p

end_time_local <- Sys.time()#end count time
print(paste("Total time on HC",as.period(round(as.duration(end_time_local-start_time_local)))))#how much
#############################################################################
# Hierarchical Clustering Phase 2
#############################################################################
start_time_local <- Sys.time()#count time
df_plot <- df_real_valued[(DIM+1):ncol(df_real_valued)]

for (k in c(1:K)) {
  data <- subset(df_plot, clusters == k) %>% 
    group_by(entity) %>% 
    count() %>% 
    ungroup() %>% 
    mutate(per=`n`/sum(`n`)) %>% 
    arrange(desc(entity))
  data$label <- scales::percent(data$per)
  p_k <- ggplot(data=data)+
    geom_bar(aes(x="", y=per, fill=entity), stat="identity", width = 1)+
    coord_polar("y", start=0)+
    labs(title = paste("K =", k))+
    theme_void()+
    theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5, size = 20))+
    fillScale+
    geom_text_repel(aes(x=1, y = cumsum(per) - per/2, label=label), size = 6)
  
  image_name <-paste("round_plot_K", k, sep = "_" )
  path_name <- paste(images_path, "round_plot", sep = "/" )
  dir.create(path_name,recursive = TRUE)
  ggsave(paste(image_name,'.png', sep = ""), plot = p_k, path = path_name, dpi = 300) # , width = 20, units = "cm"
}

end_time_local <- Sys.time()#end count time
print(paste("Total time for Phase 2",as.period(round(as.duration(end_time_local-start_time_local)))))#how much
#############################################################################
# Hierarchical Clustering Phase 3
#############################################################################
start_time_local <- Sys.time()#count time


end_time_local <- Sys.time()#end count time
print(paste("Total time for Phase 3",as.period(round(as.duration(end_time_local-start_time_local)))))#how much
#############################################################################
# SAVING FILES
#############################################################################
start_time_local <- Sys.time()#count time

# for (cluster_name in names(kmedoids)) {
#   file_name <- paste("kmedoids", cluster_name, "K" , K_list[[cluster_name]], sep = "_")
#   path_file <- paste(rds_path, "/kmedoids/", file_name, ".rds", sep = "")
#   dir.create(paste(rds_path, "/kmedoids",sep = ""),recursive = TRUE)
#   
#   saveRDS(kmedoids[[cluster_name]], file = path_file) #get(file_name)
# }

end_time_local <- Sys.time()#end count time
print(paste("Total time for saving results",as.period(round(as.duration(end_time_local-start_time_local)))))#how much
#############################################################################
end_time <- Sys.time()#end count time
print(paste("Total time for script",as.period(round(as.duration(end_time-start_time)))))#how much

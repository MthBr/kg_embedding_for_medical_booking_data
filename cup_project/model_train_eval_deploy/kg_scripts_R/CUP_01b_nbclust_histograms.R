
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
source("starting_script.R")
start_time <- Sys.time()#count time
#############################################################################
### Load data
#############################################################################
PROJECT_NAME <- "cup_1747_1" # "italy_crime_index" "cup_1747_1"

images_path <- paste(images_path, PROJECT_NAME, sep = "/")
csv_path <- paste(csv_path, PROJECT_NAME, sep = "/")
rds_path <- paste(rds_path, PROJECT_NAME, sep = "/")

df_values <- read.table(paste(csv_path, "nbclust", "df_nb_clust_values_n_start_2_end_89.csv", sep = "/"), # df_nb_clust_values_n_start_2_end_11.csv
                        header = TRUE,
                        colClasses = c(rep("character",4)),
                        sep=";")

#############################################################################
### Add libraries
#############################################################################
library(lubridate)
library(dplyr)
library(doParallel)  # will load parallel, foreach, and iterators
library(ggplot2)
library(grDevices)
library(RColorBrewer)
# source(paste(functions_path,"find_elbow.R", sep="/"))
source(paste(functions_path,"find_peaks.R", sep="/"))

#############################################################################
### B) Parallel settings
#############################################################################
#numCores <- detectCores()

##### For Windows
#cluster <- makeCluster(3)

##### For Linux
cluster <- 8
registerDoParallel(cluster)

#############################################################################
### C) Generation image and insights part
#############################################################################


histogram_path <- paste(images_path,"histograms", sep = "/")
indices_max <- c("kl","ch","ccc","silhouette","ratkowsky","ptbiserial","gamma","tau","dunn") #indices that find max values
indices_min <- c("cindex","db","mcclain","gplus","sdindex","sdbw") #indices that find min values

start_numb_clust <- 2
n_max_clusters <- length(eval(parse(text = df_values$values_index[[1]])))+(start_numb_clust-1)
x_axes <- start_numb_clust:n_max_clusters #create x-axes
df_4_hist <- setNames(data.frame(matrix(ncol = 4, nrow = 0)), c("K","distance","index_nbclust","linkage")) #INIZIALIZE

#df_4_hist = foreach (row = 1:nrow(df_values), .combine=rbind, .packages = c("ggplot2")) %do% {
for(row in 1:nrow(df_values)){
  #df_4_hist <- setNames(data.frame(matrix(ncol = 4, nrow = 0)), c("K","distance","index_nbclust","linkage")) #INIZIALIZE
  
  strNamMeth <- df_values[row, "distance"]
  linkage <-  df_values[row, "linkage"]
  nbindices <-  df_values[row, "index_nbclust"]
  
  #values <- func_c_2_num(df_values[row, "values_index"])
  values <- eval(parse(text = df_values[row, "values_index"]))
  df_plot <- data.frame(K = as.integer(x_axes),y_values = values)
  
  # values_value_max <- NULL
  # values_value_max <- find_peaks(values) #take maximums
  # names(values_value_max)<- NULL #cancel names 
  
  values_clusters <- NULL
  if(nbindices %in% indices_max){
    values_clusters <- findpeaks(values, sortstr = F, threshold = 10^-3)[,2]  #,2 is the position  #1, if sorted is the maximum!
  }else if(nbindices %in% indices_min){
    values_clusters <- findpeaks(-values, sortstr = F, threshold = 10^-3)[,2]  #,2 is the position  #1, if sorted is the maximum!
  }else next 
  
  names(values_clusters)<- NULL #cancel names 
  
  img <- ggplot(data = df_plot, aes(x=K,y=y_values))+
    geom_line()+
    geom_point()+
    labs(title = paste(nbindices ,"with", strNamMeth, linkage), y=nbindices)+  #x = "Number of Clusters"
    theme_minimal()+
    scale_x_continuous(name="Number of Clusters", breaks = x_axes) + 
    geom_point(data=df_plot[values_clusters, ], aes(x=K,y=y_values), colour="red", size=5)
  
  image_name <-paste(nbindices,"dist", strNamMeth, linkage, sep = "_" )
  
  path_name <- paste(images_path, 'nbclust', "distance", strNamMeth, sep = "/")
  dir.create(path_name,recursive = TRUE)
  ggsave(paste(image_name,'.png', sep = ""), plot = img, path = path_name, width = 20, units = "cm")
  
  path_name <- paste(images_path, "nbclust", "linkage", linkage, sep = "/" )
  dir.create(path_name,recursive = TRUE)
  ggsave(paste(image_name,'.png', sep = ""), plot = img, path = path_name, width = 20, units = "cm")
  
  path_name <- paste(images_path, "nbclust", "index_nbclust", nbindices, sep = "/" )
  dir.create(path_name,recursive = TRUE)
  ggsave(paste(image_name,'.png', sep = ""), plot = img, path = path_name, width = 20, units = "cm")
  
  
  len_values <- length(values_clusters)
  df_4_hist <- rbind(df_4_hist,data.frame("K"=values_clusters+(start_numb_clust-1),"distance"=rep(strNamMeth,len_values),"index_nbclust"=rep(nbindices,len_values),"linkage"=rep(linkage,len_values)))
}

#How to find elbow
#https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
#https://math.stackexchange.com/


generate_hists <- function(kind_of_grouping){
  
  all_unique_kind <- unique(df_4_hist[[kind_of_grouping]])
  
  for (element_kind in all_unique_kind){
    
    p2 <- ggplot(df_4_hist %>% filter(eval(parse(text = kind_of_grouping)) == element_kind), aes(x=factor(K), fill = element_kind ))+
      geom_bar(stat="count")+
      geom_text(stat='count', aes(label = ..count..),position = "stack",vjust=2)+
      theme_minimal()+
      theme(legend.position="bottom")+
      scale_fill_brewer(palette="Dark2")+
      labs(title = paste("Histogram", kind_of_grouping), x = "K")
    
    image_name <-paste("histogram", kind_of_grouping, element_kind, sep = "_" )
    path_name <- paste(histogram_path, kind_of_grouping, sep = "/" )
    dir.create(path_name,recursive = TRUE)
    ggsave(paste(image_name,'.png', sep = ""), plot = p2, path = path_name, width = 20, units = "cm")
  }
}

generate_hists("distance")

generate_hists("linkage")

generate_hists("index_nbclust")

#############################################################################
# Create histograms
#############################################################################
colourCount = max(c(length(unique(df_4_hist$distance)),length(unique(df_4_hist$index_nbclust)),length(unique(df_4_hist$linkage))))

p3 <- ggplot(df_4_hist, aes(x=factor(K), fill = distance))+ # %>% filter(K < 17)
  coord_cartesian( ylim = c(0,15)) +
  geom_bar(stat="count")+
  geom_text(stat='count', aes(label = ..count..),position = "stack",vjust=1.1,size = 2.8)+
  theme_minimal()+
  scale_fill_manual(values = colorRampPalette(brewer.pal(8, "Dark2"))(colourCount))+ #Set1
  labs(title = "Bar plot distances", x = "K")
ggsave("histogram_distance.png",  plot = p3, path = histogram_path, width = 40, dpi = "retina", units = "cm") # 40

p3 <- ggplot(df_4_hist, aes(x=factor(K), fill = index_nbclust))+ #  %>% filter(K < 17)
  coord_cartesian(ylim = c(0,15)) +
  geom_bar(stat="count")+
  geom_text(stat='count', aes(label = ..count..),position = "stack",vjust=1.1)+
  theme_minimal()+
  scale_fill_manual(values = colorRampPalette(brewer.pal(8, "Dark2"))(colourCount))+ #Set1
  labs(title = "Bar plot indices", x = "K")
ggsave("histogram_index_nbclust.png", plot = p3, path = histogram_path, width = 40, dpi = "retina", units = "cm") # 40

p3 <- ggplot(df_4_hist, aes(x=factor(K), fill = linkage))+ #  %>% filter(K < 17)
  coord_cartesian(ylim = c(0,15)) +
  geom_bar(stat="count")+
  geom_text(stat='count', aes(label = ..count..),position = "stack",vjust=1.1)+
  theme_minimal()+
  scale_fill_manual(values = colorRampPalette(brewer.pal(8, "Dark2"))(colourCount))+ #Set1
  labs(title = "Bar plot linkage methods", x = "K")
ggsave("histogram_linkage.png", plot = p3, path = histogram_path, width = 40, dpi = "retina", units = "cm") # 40

end_time <- Sys.time()#end count time
print(paste("Tempo totale",as.period(round(as.duration(end_time-start_time)))))#how much
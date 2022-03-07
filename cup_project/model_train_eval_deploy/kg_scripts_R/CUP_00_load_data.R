# install.packages('reticulate')

py_src <- "../scripts_py"  #path of python files

# Sys.setenv(RETICULATE_PYTHON = "C:\\Users\\giamp\\Anaconda3\\envs\\kg-env") # Giamp
Sys.setenv(RETICULATE_PYTHON = "/home/modal/anaconda3/envs/kg-env/bin/python") # MODAL
#Sys.setenv(RETICULATE_PYTHON = "/home/enzo/miniconda3/envs/kg-env/bin/python") # Enzo
#Sys.setenv(RETICULATE_PYTHON = "/home/.enzoDpbFld/anaconda3/envs/kg-env/bin/python")


require("reticulate")
py_config()
#env!
#use_condaenv("kg-env", required = TRUE)
#py_config()


#SET setting it in ~/.Renviron
#with RETICULATE_PYTHON="/envs/kg-env/bin/python" (
#Sys.setenv(RETICULATE_PYTHON = "/home/enzo/miniconda3/envs/kg-env/bin/python")
# R.home()
#file.edit(".Rprofile")
#file.edit(file.path("/usr/lib/R", ".Rprofile"))

#detach("package:reticulate", unload=TRUE)
#require("reticulate")

#use_python("envs/kg-env/bin/python")
#use_condaenv("kg-env", required = TRUE)
py_config()



#py_file <- paste(py_src, "from_embedding_to_clustering.py", sep = "/")
#source_python(py_file, convert = TRUE) 
py_modul <- import_from_path("from_embedding_to_clustering", path = py_src, convert = TRUE)
importlib <- import("importlib")
importlib$reload(py_modul)

#emb <- py_modul$dump_full_embedding()

#concept_emb <-py_modul$get_dict("cup_1747_1", "embeddings")  #cup_100_1
PROJECT_NAME <- "cup_1747_1"

df_real_valued <- py_modul$get_emb(PROJECT_NAME, "concepts")  #cup_100_1
DIM <- 100*round(ncol(df_real_valued)/100)


#install.packages('reticulate')

py_src <- "../scripts_py"  #path of python files

#Sys.setenv(RETICULATE_PYTHON = "/home/enzo/miniconda3/envs/kg-env/bin/python")
Sys.setenv(RETICULATE_PYTHON = "/home/.enzoDpbFld/anaconda3/envs/kg-env/bin/python")


require("reticulate")
py_config()
py_config()


py_modul <- import_from_path("model_analysis", path = py_src, convert = TRUE)

#emb <- py_modul$dump_full_embedding()


models_df <- py_modul$get_model_df(file_name="cup_100_1_t1.log")












# This file is generated and can be overwritten.
 library(dplyr) 

refine_dataframe <- function(df) { 
  df <- mutate( df ,`CHURN` = as.logical(`CHURN`))
  df <- mutate( df ,`Children` = as.integer(`Children`))
  df <- mutate( df ,`Est Income` = as.double(`Est Income`))
  df <- mutate( df ,`Age` = as.double(`Age`))
  return (df) 
}

refine_file <- function(inputFile, outputFile) { 
  if(file.exists(inputFile)){ 
    if (!is.na(outputFile) && file.exists(outputFile)) {
      return(paste0("Output file: ", outputFile, " already exists "))
    }
    df <- read.csv(inputFile, check.names=FALSE) 
    df <- refine_dataframe(df) 
    if (!is.na(outputFile)) {
        write.csv(df, file = outputFile, row.names=FALSE)
        print(paste0(paste0("Writing to ", outputFile) ," file is complete"))
    } else {
      return (df)
    }
  } else {
    print(paste0(inputFile, " file does not exist ")) 
   }
} 

args <- commandArgs(TRUE)
if (length(args)>0) { 
  refine_file(args[1], args[2])
}
library(affy)
library(dplyr)
library(data.table)

MAIN_DIR = '.'
NORMALIZING = 'YES' #['YES', 'NO'] if normalizing for the first time select YES, otherwise select NO
setwd(MAIN_DIR) # change working directory 
# CEL files
DATASETS = list('GSE11121',
                'GSE18864',
                'GSE20711', 
                'GSE23593',
                'GSE27120',
                'GSE32646',
                'GSE36771', 
                'GSE42568',
                'GSE50948',
                'GSE5460',
                'GSE11001',
                'GSE87007',
                'GSE88770',
                'GSE7390', 
                'GSE78958',
                'GSE45255',
                'GSE61304',
                'GSE63471',
                'GSE21653',
                'GSE26639',
                'GSE17907',
                'GSE10810',
                'GSE25066',
                'GSE47109',
                'GSE95700',
                'GSE5327',
                'GSE48390',
                'GSE58984',
                'GSE103091',
                'GSE45827',
                'GSE65194',
                'GSE1456',
                'GSE102484')

for (dataset in DATASETS){   
    # loading the dataset 
    env <- new.env()
    nm <- load(sprintf('%s/%s/%s_Affy.RData', MAIN_DIR, dataset, dataset), env)[1]
    cancer_data <- env[[nm]]

    if (NORMALIZING == 'YES'){

        # Creating working directory rma
        dir.create(sprintf('%s/%s/rma',MAIN_DIR, dataset))

        # rma
        cancer_data.rma <- rma(cancer_data)

        # save rma (pancreatic.rma.RData)
        save(cancer_data.rma , file= sprintf('%s/%s/rma/%s.rma.RData', MAIN_DIR, dataset, dataset))
        cancer_data.rma
    } else{

        # loading dataset '* _Affy.RData'--- from 1. R_Cell_files_ to _affy
        env <- new.env()
        nm <- load(sprintf('%s/%s/rma/%s.rma.RData', MAIN_DIR, dataset, dataset), env)[1]
        cancer_data.rma <- env[[nm]]

    }

    cancer_data.matrix <- affy::exprs(cancer_data.rma)
    head(cancer_data.matrix)

    # save expression as RData (.matrix.RData)
    save(cancer_data.matrix, file=sprintf('%s/%s/rma/%s.matrix.RData', MAIN_DIR, dataset, dataset))

    # save exptession as csv (.matrix.csv)
    write.csv(cancer_data.matrix, file= sprintf('%s/%s/rma/%s.matrix.csv', MAIN_DIR, dataset, dataset), row.names=TRUE)}

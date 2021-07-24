library(affy)
library(inSilicoMerging)

MAIN_DIR = '.'
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

# Changeing the cels data to Affy to use in rma (ReadAffy)
esets = list()
for(dataset in DATASETS) {
    env <- new.env()
    filename = sprintf('%s/main_data/%s/rma/%s.rma.RData', MAIN_DIR, dataset, dataset)
    print(sprintf('Loading file %s', filename))
    nm <- load(filename, env)[1]
    cancer_data <- env[[nm]]
    pData(cancer_data)['sample'] = dataset
    esets <- append(esets, cancer_data)
}

eset_COMBAT = inSilicoMerging::merge(esets, method="COMBAT")

# saveing affy file 
dir.create(sprintf('%s/analysed_datasets/merged', MAIN_DIR))
save(eset_COMBAT, file=sprintf('%s/analysed_datasets/merged/eset_merged_COMBAT_rma.RData', MAIN_DIR))

cancer_data.matrix <- affy::exprs(eset_COMBAT)
head(cancer_data.matrix)

# list of dublicated samples
len = ncol(cancer_data.matrix)
duplicated_names = list()
sample_names = colnames(cancer_data.matrix)
j = 0
for(i in 1:len){
    item = unlist(strsplit(unlist(strsplit(colnames(cancer_data.matrix)[i], split='.', fixed=TRUE))[1], split='_', fixed=TRUE))[1]
    
    if(!item %in% sample_names){
      sample_names[i] = item
    }
    else {
        duplicated_names[j] = sample_names[i]
        j = j + 1
   }
    
}

duplicated_names

# remove dublicate samples
len = ncol(cancer_data.matrix)
duplicated_names = list()
sample_names = colnames(cancer_data.matrix)
j = 0

for(i in 1:len){
    item = unlist(strsplit(unlist(strsplit(colnames(cancer_data.matrix)[i], split='.', fixed=TRUE))[1], split='_', fixed=TRUE))[1]
    sample_names[i] = item  
}

colnames(cancer_data.matrix)= sample_names

# save exptession as csv (.matrix.csv)
write.csv(cancer_data_uniq.matrix, file= sprintf('%s/analysed_datasets/merged/merged_COMBAT_rma.matrix.csv',MAIN_DIR), row.names=TRUE)

plotMDS(eset_COMBAT)

# This code is used to generate Spearman correlation valueas 
# stored in 'coreelation_matrix'
data <- read.csv("/share_large/lbcg/data/yeast/", row.names = 1)

correlation_matrix <- cor(t(data), method="spearman")

write.csv(correlation_matrix, "correlation_matrix.csv")
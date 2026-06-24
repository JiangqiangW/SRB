setwd("C:\\R\\ ")
library("qgraph")
library("networktools")
library("ggplot2")
library("bootnet")
library("grDevices")
library("psych")
library("dplyr")
library("tidyverse")
library("showtext")
library("readxl")
library("boot")
library("mgm")    
library(openxlsx)

# ============================================================
# Loading data
# ============================================================
mydata <- read_excel("SRB.xlsx")

# ============================================================
# Variable selection
# ============================================================
vars <- c("S_time", "S_anx1", "S_anx2", "S_anx3", "EBB3", "R_tea", "Inattention", "LC", "AC", "WE", "PL")

data <- mydata[, vars]

# ============================================================
# network estimate
# ============================================================

#  cor_auto 
CorMat <- cor_auto(data)

# Node predictability
data_matrix <- as.matrix(data)
p <- ncol(data_matrix)
fit_obj <- mgm(
  data   = data_matrix,
  type   = rep('g', p),
  level  = rep(1, p),
  lambdaSel = 'CV',
  ruleReg   = 'OR',
  pbar      = FALSE
)
pred_obj <- predict(fit_obj, data_matrix)


Network <- estimateNetwork(
  data,
  default   = "EBICglasso",
  tuning    = 0.5,
  corMethod = "cor_auto"    
)

# Network structure
pdf(file = 'network_vars_exploratory.pdf', width = 10, height = 7)

qgraph(
  Network$graph,
  layout    = "spring",
  vsize     = 9,
  color     = "lightblue",
  pie       = pred_obj$error[, 2],
  label.cex = 1.2,
  legend.cex = 0.5
)

dev.off()

# ============================================================
# Centrality metric
# ============================================================

# Computing Centrality
centrality <- centrality_auto(Network$graph, weighted = TRUE, signed = TRUE)
nc <- centrality$node.centrality
print(nc)
write.csv(nc, file = "centrality_vars.csv")

# Centralization Visualization
pdf(file = 'centrality_plot_vars.pdf', width = 4, height = 7)
centralityPlot(
  Network$graph,
  include = c("ExpectedInfluence"),scale = "z-scores",orderBy="ExpectedInfluence"
)
dev.off()

# ============================================================
#  Bootstrap
# ============================================================

# case-dropping bootstrap
boot_case <- bootnet(
  Network,
  statistics = c("ExpectedInfluence"),
  nBoot      = 1000,
  nCore      = 8,
  type       = "case"
)

pdf(file = 'stability_centrality_EI.pdf', width = 10, height = 7)
plot(boot_case, statistics = c("ExpectedInfluence"))
dev.off()


# Extract edge weight matrix
edge_matrix <- getWmat(Network$graph)

write.xlsx(edge_matrix, file = "edge_weights_matrix.xlsx", rowNames = TRUE)

# CS 
cat("\n===== CS coefficient =====\n")
corStability(boot_case)

# ============================================================
# 5.2 Edge weight Bootstrap confidence interval
# ============================================================
boot_edge <- bootnet( Network,statistics = c("edge"), nBoot = 1000, nCore = 8)

pdf(file = 'edge_bootstrap_7vars.pdf', width = 4, height = 7)
plot(boot_edge, labels = FALSE, order = "sample")
dev.off()


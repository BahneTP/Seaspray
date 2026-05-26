options(scipen = 999)

suppressPackageStartupMessages(library(mclust))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript mclust_gmm.R <dataset.csv> <output.csv>", call. = FALSE)
}

dataset_path <- args[[1]]
output_path <- args[[2]]
features <- c("x1", "x2")

df <- read.csv(dataset_path)
train_df <- df[df$cluster %in% c("A", "B"), ]
#train_df <- df[df$cluster %in% c("A", "B", "middle"), ]

X_train <- as.matrix(train_df[, features])
X_all <- as.matrix(df[, features])

X_train_scaled <- scale(X_train)
center <- attr(X_train_scaled, "scaled:center")
scale_ <- attr(X_train_scaled, "scaled:scale")
X_all_scaled <- scale(X_all, center = center, scale = scale_)

gmm <- Mclust(
  X_train_scaled,
  G = 2,
  #modelNames = "VVV",
  verbose = FALSE
)
if (is.null(gmm)) {
  gmm <- Mclust(
    X_train_scaled,
    G = 2,
    verbose = FALSE
  )
}
if (is.null(gmm)) {
  stop("mclust could not estimate a GMM for this dataset", call. = FALSE)
}

cat("Selected mclust modelName:", gmm$modelName, "\n")

train_pred <- predict(gmm, newdata = X_train_scaled)
a_means <- colMeans(train_pred$z[train_df$cluster == "A", , drop = FALSE])
cluster_a_component <- which.max(a_means)

pred <- predict(gmm, newdata = X_all_scaled)

result <- data.frame(
  row_index = seq_len(nrow(df)) - 1,
  mclust_prob_a = pred$z[, cluster_a_component],
  mclust_model = gmm$modelName
)

write.csv(result, output_path, row.names = FALSE)

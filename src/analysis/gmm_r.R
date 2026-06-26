options(scipen = 999)

suppressPackageStartupMessages(library(mclust))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop(
    "Usage: Rscript gmm_r.R <train.csv> <all.csv> <output.csv> <feature_1> [feature_2 ...]",
    call. = FALSE
  )
}

train_path <- args[[1]]
all_path <- args[[2]]
output_path <- args[[3]]
features <- args[4:length(args)]

train_df <- read.csv(train_path, check.names = FALSE)
all_df <- read.csv(all_path, check.names = FALSE)

X_train <- as.matrix(train_df[, features])
X_all <- as.matrix(all_df[, features])

X_train_scaled <- scale(X_train)
center <- attr(X_train_scaled, "scaled:center")
scale_ <- attr(X_train_scaled, "scaled:scale")
X_all_scaled <- scale(X_all, center = center, scale = scale_)

gmm <- Mclust(
  X_train_scaled,
  G = 3,
  verbose = FALSE
)

if (is.null(gmm)) {
  gmm <- Mclust(
    X_train_scaled,
    G = 3,
    verbose = FALSE
  )
}

if (is.null(gmm)) {
  stop("mclust could not estimate a GMM for this dataset", call. = FALSE)
}

pred <- predict(gmm, newdata = X_all_scaled)

result <- data.frame(
  row_index = seq_len(nrow(all_df)) - 1,
  gmm_prob_0 = pred$z[, 1],
  gmm_prob_1 = pred$z[, 2],
  gmm_prob_2 = pred$z[, 3],
  gmm_model = gmm$modelName
)

write.csv(result, output_path, row.names = FALSE)

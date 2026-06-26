from gmm_r import GMM_R


def run_gmm(df_train, df_eval, feature_cols):
    gmm = GMM_R(df_train, feature_cols)
    return gmm.save(df_eval)

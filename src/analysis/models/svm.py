from svm import SVM


def run_svm(df_train, df_eval, feature_cols):
    svm = SVM(df_train, df_eval, feature_cols)
    return svm.save(df_eval)

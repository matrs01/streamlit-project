import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

CATEGORICAL_FEATURES = ['EDUCATION', 'MARITAL_STATUS', 'SOCSTATUS_WORK_FL',
                        'SOCSTATUS_PENS_FL', 'WORK_STATUS', 'PENSION_STATUS']


def numerical_feature_distribution(processed_df, features=['All']):
    # Setting the aesthetic style of the plots
    sns.set(style="whitegrid")

    # Selecting specific columns for adjusted binning
    columns_plot_info = {
        'PERSONAL_INCOME': {'bin_step': 2000, 'kde': False},
        'CREDIT': {'bin_step': 'auto', 'kde': True},
        'AGE': {'bin_step': 1, 'kde': True},
        'CHILD_TOTAL': {'bin_step': 0.25, 'kde': False},
        'GENDER': {'bin_step': 'auto', 'kde': False},
        'DEPENDANTS': {'bin_step': 0.25, 'kde': False},
        'TERM': {'bin_step': 4, 'kde': False},
        'LOAN_NUM_TOTAL': {'bin_step': 0.25, 'kde': False},
        'LOAN_NUM_CLOSED': {'bin_step': 0.25, 'kde': False},
        'OWN_AUTO': {'bin_step': 'auto', 'kde': False},
        'TARGET': {'bin_step': 'auto', 'kde': False},
        'FST_PAYMENT': {'bin_step': 800, 'kde': True},
    }

    if 'All' in features:
        features = list(columns_plot_info.keys())

    nrows = (len(features) + 2) // 3
    ncols = 3
    fig = plt.figure(figsize=(15, 5 * nrows))

    for i in range(len(features)):
        data = processed_df[features[i]]

        bin_step = columns_plot_info[features[i]]['bin_step']
        bins = np.arange(data.min(), data.max(),
                         bin_step) if bin_step != 'auto' else 'auto'
        kde = columns_plot_info[features[i]]['kde']

        ax = plt.subplot(nrows, ncols, i + 1)
        sns.histplot(data, bins=bins, kde=kde, ax=ax)
        plt.title(f'Distribution of {features[i]}')
        plt.xlabel(features[i])
        plt.ylabel('Count')

    plt.tight_layout()

    return fig


def categorical_feature_distribution(processed_df, features=['All']):
    def adjust_label_positions_for_horizontal(ax):
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')
            label.set_rotation(45)

    # Setting the aesthetic style of the plots
    sns.set(style="whitegrid")

    if 'All' in features:
        features = CATEGORICAL_FEATURES

    nrows = (len(features) + 2) // 3
    ncols = 3
    fig = plt.figure(figsize=(15, 5 * nrows))

    for i in range(len(features)):
        ax = plt.subplot(nrows, ncols, i + 1)
        sns.countplot(x='EDUCATION', data=processed_df, ax=ax)
        plt.title(f'Distribution of {features[i]}')
        adjust_label_positions_for_horizontal(ax)
        plt.xlabel('')

    plt.tight_layout()

    return fig


def correlation_map(processed_df):
    # Removing the 'ID_CLIENT' and 'ID_LOAN' columns which are identifiers
    df_reduced = processed_df.drop(
        columns=['ID_CLIENT', 'ID_LOAN', 'AGREEMENT_RK'])

    # Selecting numerical columns again for the correlation matrix
    numerical_cols_reduced = df_reduced.select_dtypes(
        include=['float64', 'int64']).columns
    correlation_matrix_reduced = df_reduced[numerical_cols_reduced].corr()

    # Plotting the final correlation matrix with adjusted label alignment
    fig = plt.figure(figsize=(17, 15))
    sns.heatmap(correlation_matrix_reduced, annot=True,
                fmt=".1f", cmap='coolwarm', linewidths=.5)

    # Adjusting the x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.title("Final Correlation Matrix with Adjusted Labels")

    return fig

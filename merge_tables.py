import pandas as pd
import paths

# Load the dataframes
df_clients = pd.read_csv(paths.CLIENTS_PATH)
df_target = pd.read_csv(paths.TARGET_PATH)
df_job = pd.read_csv(paths.JOB_PATH)
df_salary = pd.read_csv(paths.SALARY_PATH)
df_last_credit = pd.read_csv(paths.LAST_CREDIT_PATH)
df_loan = pd.read_csv(paths.LOAN_PATH)
df_close_loan = pd.read_csv(paths.CLOSE_LOAN_PATH)
df_work = pd.read_csv(paths.WORK_PATH)
df_pens = pd.read_csv(paths.PENS_PATH)

# Merging the dataframes
# The common key seems to be 'ID_CLIENT' for most dataframes and 'ID' for D_clients
# First, rename 'ID' in D_clients to 'ID_CLIENT' for consistency
df_clients = df_clients.rename(columns={"ID": "ID_CLIENT"})

# Merge dataframes on 'ID_CLIENT'
merged_df = df_clients.merge(df_target, on='ID_CLIENT', how='left')\
                      .merge(df_job, on='ID_CLIENT', how='left')\
                      .merge(df_salary, on='ID_CLIENT', how='left')\
                      .merge(df_last_credit, on='ID_CLIENT', how='left')\
                      .merge(df_loan, on='ID_CLIENT', how='left')\
                      .merge(df_close_loan, on='ID_LOAN', how='left')\
                      .merge(df_work, left_on='SOCSTATUS_WORK_FL', right_on='FLAG', how='left')\
                      .merge(df_pens, left_on='SOCSTATUS_PENS_FL', right_on='FLAG', how='left')


# Processing the merged dataframe

# Removing redundant columns and renaming columns for clarity
columns_to_remove = ['ID_x', 'ID_y', 'FLAG_x', 'FLAG_y']
columns_to_rename = {
    'COMMENT_x': 'WORK_STATUS',
    'COMMENT_y': 'PENSION_STATUS'
}

# Applying the changes
processed_df = merged_df.drop(
    columns=columns_to_remove).rename(columns=columns_to_rename)

# Additionally, check for any duplicate rows based on 'ID_CLIENT'
processed_df = processed_df.drop_duplicates()

# drop raws where TARGET is Nan
processed_df = processed_df.dropna(subset=['TARGET'])

# Check for missing values in key columns
# missing_values_summary = processed_df.isnull().sum()

# Handling missing values in numerical columns

numerical_cols = ['WORK_TIME']

# Filling missing values with the median (a common approach for numerical data)
for col in numerical_cols:
    processed_df[col].fillna(processed_df[col].median(), inplace=True)

# Now, handling missing values in categorical columns
categorical_cols = ['GEN_INDUSTRY', 'GEN_TITLE', 'JOB_DIR']

# Filling missing values with the mode (most frequent value)
for col in categorical_cols:
    processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)

# Recheck for missing values
# missing_values_after_handling = processed_df.isnull().sum()
# Output: no missing values

# calculating LOAN_NUM_TOTAL and LOAN_NUM_CLOSED

# Counting the total number of loans (ID_LOAN) for each client (ID_CLIENT)
loan_num_total = processed_df.groupby(
    'ID_CLIENT')['ID_LOAN'].nunique().reset_index(name='LOAN_NUM_TOTAL')

# Counting the number of closed loans for each client
# A loan is considered closed if CLOSED_FL equals 1
loan_num_closed = processed_df[processed_df['CLOSED_FL'] == 1].groupby(
    'ID_CLIENT')['ID_LOAN'].nunique().reset_index(name='LOAN_NUM_CLOSED')


# Then, we merge the new columns
processed_df = processed_df.merge(loan_num_total, on='ID_CLIENT', how='left')
processed_df = processed_df.merge(loan_num_closed, on='ID_CLIENT', how='left')

# Filling any NaN values in LOAN_NUM_CLOSED with 0 (assuming NaN means no closed loans)
processed_df['LOAN_NUM_CLOSED'].fillna(0, inplace=True)

# Saving processed dataframe to csv
processed_df.to_csv(paths.PROCESSES_DATA_DIR, index=False)

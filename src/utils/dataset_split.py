import pandas as pd
import numpy as np

# Ground truth input file paths
train_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_ground-truth_dev-set.csv'
test_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/test_ground-truth_dev-set.csv'

# Ground truth output file paths
train_ground_truth_splitted_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_ground-truth_dev-set_splitted.csv'
test_ground_truth_splitted_file_path = '../../data/corpus/devset/dev-set/ground-truth/test_ground-truth_dev-set_splitted.csv'

# Captions input file paths
train_captions_file_path = '../../data/corpus/devset/dev-set/train_dev-set_video-captions-cleanup.csv'
test_captions_file_path = '../../data/corpus/devset/dev-set/test_dev-set_video-captions-cleanup.csv'

# Captions output file paths
train_captions_splitted_file_path = '../../data/corpus/devset/dev-set/train_dev-set_video-captions-cleanup_splitted.csv'
test_captions_splitted_file_path = '../../data/corpus/devset/dev-set/test_dev-set_video-captions-cleanup_splitted.csv'

# load dataframes
df_train_ground_truth = pd.read_csv(train_ground_truth_file_path)
df_test_ground_truth = pd.read_csv(test_ground_truth_file_path)

df_train_captions = pd.read_csv(train_captions_file_path)
df_test_captions = pd.read_csv(test_captions_file_path)

# order columns by long-term_memorability
# We have chosen long term memorability because it has a lower mean and higher variance
df_train_ground_truth = df_train_ground_truth.sort_values(by=['long-term_memorability'], ascending=False)
df_test_ground_truth = df_test_ground_truth.sort_values(by=['long-term_memorability'], ascending=False)

# Now that it is ordered, split it.
train_higher_values, train_middle_values, train_lower_values = np.split(df_train_ground_truth, [int(0.25*len(df_train_ground_truth)), int(0.75*len(df_train_ground_truth))])
test_higher_values, test_middle_values, test_lower_values = np.split(df_test_ground_truth, [int(0.25*len(df_test_ground_truth)), int(0.75*len(df_test_ground_truth))])

# Merge higher values and lower values
df_train_ground_truth_splitted = pd.concat([train_higher_values, train_lower_values])
df_test_ground_truth_splitted = pd.concat([test_higher_values, test_lower_values])

# Now split the captions dataset too.
df_train_captions_splitted = df_train_captions[df_train_captions.id.isin(df_train_ground_truth_splitted.video)]
df_test_captions_splitted = df_test_captions[df_test_captions.id.isin(df_test_ground_truth_splitted.video)]

# Sort them
df_train_ground_truth_splitted = df_train_ground_truth_splitted.sort_values(by=['video'])
df_test_ground_truth_splitted = df_test_ground_truth_splitted.sort_values(by=['video'])

df_train_captions_splitted = df_train_captions_splitted.sort_values(by=['id'])
df_test_captions_splitted = df_test_captions_splitted.sort_values(by=['id'])

# Finally, save the new datasets
print ('Ground truth splitted set [train=' + str(len(df_train_ground_truth_splitted)) + ",test=" + str(len(df_test_ground_truth_splitted)) + "]")
print ('Captions splitted set [train=' + str(len(df_train_captions_splitted)) + ",test=" + str(len(df_test_captions_splitted)) + "]")

df_train_ground_truth_splitted.to_csv(train_ground_truth_splitted_file_path, index=False)
df_test_ground_truth_splitted.to_csv(test_ground_truth_splitted_file_path, index=False)

df_train_captions_splitted.to_csv(train_captions_splitted_file_path, index=False)
df_test_captions_splitted.to_csv(test_captions_splitted_file_path, index=False)
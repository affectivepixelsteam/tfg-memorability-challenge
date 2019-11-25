# So here what we are going to do is to split our dev-set in test and train sets. Since our video set is so large what we
# are actually going to do is to split the ground-truth csv so anytime we want to get our train test just take the videos
# that are contained in the train_ground-truth.csv. Same happens with test_ground-truth.
import pandas as pd
from sklearn.model_selection import train_test_split 

# So we are going to get the train and test set from the splitted dataset and the whole dataset.
ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/ground-truth_dev-set.csv'
captions_file_path = '../../data/corpus/devset/dev-set/dev-set_video-captions-cleanup.csv'

# Split params
test_size = 0.2
round_state = 123


# Ground truth output file paths
train_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_ground-truth_dev-set.csv'
test_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/test_ground-truth_dev-set.csv'

# Captions output file paths
train_captions_file_path = '../../data/corpus/devset/dev-set/train_dev-set_video-captions-cleanup.csv'
test_captions_file_path = '../../data/corpus/devset/dev-set/test_dev-set_video-captions-cleanup.csv'


# Get dataframes.
df_ground_truth = pd.read_csv(ground_truth_file_path)
df_captions = pd.read_csv(captions_file_path)

# Split ground_truths
train_ground_truth_all, test_ground_truth_all = train_test_split(df_ground_truth, test_size=test_size, random_state=round_state)

# Do the same split on captions files
train_captions_all = df_captions[df_captions.id.isin(train_ground_truth_all.video)]
test_captions_all = df_captions[df_captions.id.isin(test_ground_truth_all.video)]

# Sort values before saving them.
train_ground_truth_all = train_ground_truth_all.sort_values(by=['video'])
test_ground_truth_all = test_ground_truth_all.sort_values(by=['video'])

train_captions_all = train_captions_all.sort_values(by=['id'])
test_captions_all = test_captions_all.sort_values(by=['id'])

# Save files

print ('Ground truth set [train=' + str(len(train_ground_truth_all)) + ",test=" + str(len(test_ground_truth_all)) + "]")
print ('Captions set [train=' + str(len(train_captions_all)) + ",test=" + str(len(test_captions_all)) + "]")

train_ground_truth_all.to_csv(train_ground_truth_file_path, index=False)
test_ground_truth_all.to_csv(test_ground_truth_file_path, index=False)

train_captions_all.to_csv(train_captions_file_path, index=False)
test_captions_all.to_csv(test_captions_file_path, index=False)

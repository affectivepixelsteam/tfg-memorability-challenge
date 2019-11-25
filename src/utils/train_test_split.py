# So here what we are going to do is to split our dev-set in test and train sets. Since our video set is so large what we
# are actually going to do is to split the ground-truth csv so anytime we want to get our train test just take the videos
# that are contained in the train_ground-truth.csv. Same happens with test_ground-truth.
import pandas as pd
from sklearn.model_selection import train_test_split 

# So we are going to get the train and test set from the splitted dataset and the whole dataset.
ground_truth_splitted_file_path = '../../data/corpus/devset/dev-set/ground-truth/ground-truth_dev-set_splitted.csv'
ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/ground-truth_dev-set.csv'

# Output file paths
train_ground_truth_splitted_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_ground-truth_dev-set_splitted.csv'
test_ground_truth_splitted_file_path = '../../data/corpus/devset/dev-set/ground-truth/test_ground-truth_dev-set_splitted.csv'
train_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_ground-truth_dev-set.csv'
test_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/test_ground-truth_dev-set.csv'

# Get dataframes.
splitted_df = pd.read_csv(ground_truth_splitted_file_path)
all_df = pd.read_csv(ground_truth_file_path)

# Split ground_truths
train_splitted, test_splitted = train_test_split(splitted_df, test_size=0.2, random_state=123)
train_all, test_all = train_test_split(all_df, test_size=0.2, random_state=123)

# Save files
train_splitted.to_csv(train_ground_truth_splitted_file_path, index=False)
test_splitted.to_csv(test_ground_truth_splitted_file_path, index=False)
train_all.to_csv(train_ground_truth_file_path, index=False)
test_all.to_csv(test_ground_truth_file_path, index=False)

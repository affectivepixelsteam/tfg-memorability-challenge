import pandas as pd
import numpy as np

# Path to files/folders
ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/ground-truth_dev-set.csv'
captions_file_path = '../../data/corpus/devset/dev-set/dev-set_video-captions-cleanup.csv'

splitted_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/ground-truth_dev-set_splitted.csv'
splitted_captions_file_path = '../../data/corpus/devset/dev-set/dev-set_video-captions-cleanup_splitted.csv'

# load dataframes
df_ground_truth = pd.read_csv(ground_truth_file_path)
df_captions = pd.read_csv(captions_file_path)

# order columns by long-term_memorability
# We have chosen long term memorability because it has a lower mean and higher variance
df_ground_truth = df_ground_truth.sort_values(by=['long-term_memorability'], ascending=False)

# Now that it is ordered, split it.
higher_values, middle_values, lower_values = np.split(df_ground_truth, [int(0.25*len(df_ground_truth)), int(0.75*len(df_ground_truth))])

# Merge higher values and lower values
df_ground_truth_splitted = pd.concat([higher_values, lower_values])

# Now split the captions dataset too.
df_captions_splitted = df_captions.iloc[df_ground_truth_splitted.index.values.tolist(),:]

# Finally, save the new datasets
df_ground_truth_splitted.to_csv(splitted_ground_truth_file_path, index=False)
df_captions_splitted.to_csv(splitted_captions_file_path, index=False)
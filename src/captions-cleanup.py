# Cleanup captions dev-set and transform it to a csv file.

# Path of captions file
path = '../corpus/devset/dev-set/dev-set_video-captions.txt'

# Path to write cleanup file
path_to_write = '../corpus/devset/dev-set/dev-set_video-captions-cleanup.csv'

# Open file
with open (path, 'r', encoding="utf-8") as captions_file:

    # Read all lines and store them as a list
    captions_text = captions_file.readlines()

    # String to write in file
    final_string = ''
    
    for line in captions_text:

        # Split line to get each video id and caption
        captions_text_splitted = line.split('\t')

        # Retrieve video id and caption
        video_id = captions_text_splitted[0]
        caption_text = captions_text_splitted[1]

        # Remove '-' from captions.
        caption_text = ' '.join(caption_text.split('-'))

        # Store them in array
        final_string += video_id.strip() + ',' + caption_text.strip() + "\n"


with open(path_to_write, 'w', encoding="utf-8") as cleanup_file:
    # Columns
    cleanup_file.writelines('id,captions' + '\n')
    # All captions.
    cleanup_file.write(final_string) 
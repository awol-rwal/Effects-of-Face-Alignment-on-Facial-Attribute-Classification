import shutil
import os

# Alter the following values
SOURCE_PATH = "face_aligned_images"
TRAIN_PATH = "aligned_train"
VALIDATE_PATH = "aligned_validate"
TEST_PATH = "aligned_test"

# Make sure to update all necessary file and directory paths
if __name__ == "__main__":
    
    # read from parition.txt
    partition_fobj = open("face_align_list_partition.txt", "r")
    labels_fobj = open("face_align_list_attr.txt", "r")
    image_path = SOURCE_PATH
    # date
    labels_fobj.readline().strip().split()
    # label names
    label_names = labels_fobj.readline()

    # Clearing folder contents
    directories = [TRAIN_PATH, VALIDATE_PATH, TEST_PATH]
    for directory in directories:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.remove(item_path)  # Remove files
                print(f"Removed {item_path}")

    # Create label files
    open(f'{TRAIN_PATH}/labels.txt', 'w').close()
    open(f'{VALIDATE_PATH}/labels.txt', 'w').close()
    open(f'{TEST_PATH}/labels.txt', 'w').close()

    train_labels_fobj = open(f"{TRAIN_PATH}/labels.txt", "a")
    validate_labels_fobj = open(f"{VALIDATE_PATH}/labels.txt", "a")
    test_labels_fobj = open(f"{TEST_PATH}/labels.txt", "a")
    train_labels_fobj.write(label_names)
    validate_labels_fobj.write(label_names)
    test_labels_fobj.write(label_names)

    counter = 0
    # for each line 
    while True:
        # read file name, partition and label values
        partition = partition_fobj.readline().strip().split()
        label = labels_fobj.readline()
        if not label:
            break
        # if 0 == train, 1==validate, 2==test
        if partition[1] == "0":
            path = TRAIN_PATH
            # write label to corresponding file
            train_labels_fobj.write(label)
        elif partition[1] == "1":
            path = VALIDATE_PATH
            validate_labels_fobj.write(label)
        elif partition[1] == "2":
            path = TEST_PATH
            test_labels_fobj.write(label)

        # move image in images to corresponding folder
        shutil.copy(f"{image_path}/{partition[0]}", f"{path}/{partition[0]}")
        print(f"Moved {partition[0]} to {path}")
        counter += 1




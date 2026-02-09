import os
import shutil
import random
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True)
parser.add_argument('--dataset_dir', default='/home/higashi/symlink_hdd/higashi/dataset')
parser.add_argument('--data_num', default=200)



args = parser.parse_args()

dataset_dir = args.dataset_dir
if args.mode == 'rgb':
    opt = '.avi'
    exclude = '_rgb.avi'
    rturgb_dir = '/home/higashi/symlink_hdd/higashi/nturgb+d_rgb'
elif args.mode == 'depth':
    opt = ''
    exclude = ''
    rturgb_dir = '/home/higashi/symlink_hdd/higashi/nturgb+d_depth_masked'
elif args.mode == 'skeleton':
    opt = '.pkl'
    exclude = '.pkl'
    rturgb_dir = '/home/higashi/symlink_hdd/higashi/nturgb+d_skeletons_pkl'
    
train_txt_path = os.path.join(dataset_dir, 'train.txt')
test_txt_path = os.path.join(dataset_dir, 'test.txt')

binary = False
data_num = args.data_num
train_rate = 0.7
random.seed(923)

os.makedirs(dataset_dir, exist_ok=True)

if binary:
    ## binary
    targets = ['A043']
    # others = [f'A{i:03}' for i in range(120)]
    others = ['A008', 'A009', 'A006','A007', 'A024', 'A027','A099','A100','A044','A041']
    others = list(set(targets) ^ set(others))
else:
    ## multi class
    # targets = ['A008', 'A009', 'A043' ,'A006','A007', 'A024', 'A027','A099','A100','A044']
    targets = ['A008',  'A043' ,'A006', 'A009','A044','A050']#MAD
    # targets = ['A008', 'A009', 'A043']
    # targets = ['A006','A008','A050','A043']
    others = []

    classes = sorted(targets); assert len(others) == 0, 'If you use multi-class, "others" should be []' 

train_paths, test_paths = [], []
for tgt in targets:
    tgt_paths = glob.glob(os.path.join(rturgb_dir, f'*{tgt}*{opt}'))
    random.shuffle(tgt_paths)
    tgt_paths = tgt_paths[:data_num]
    train_paths += tgt_paths[:int(len(tgt_paths)*train_rate)]
    test_paths += tgt_paths[int(len(tgt_paths)*train_rate):]
for other in others:
    other_paths = glob.glob(os.path.join(rturgb_dir, f'*{other}*{opt}'))
    random.shuffle(other_paths)
    other_paths = other_paths[:data_num]
    train_paths += other_paths[:int(len(other_paths)*train_rate)]
    test_paths += other_paths[int(len(other_paths)*train_rate):]

random.shuffle(train_paths)
random.shuffle(test_paths)

for paths, txt_path in zip([train_paths, test_paths], [train_txt_path, test_txt_path]):
    with open(txt_path, 'w') as f:
        for path in paths:
            label_name = 'A' + os.path.basename(path).split('A')[-1].rstrip(exclude)
            if binary:
                label = int(label_name in targets)
            else:
                label = classes.index(label_name)
            f.write(f'{path} {label}\n')

 
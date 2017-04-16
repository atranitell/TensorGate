
import re
import gate


def write_checkpoint_model_items(chkp_file_path, chkp_model_list):
    """ Write model items to checkpoint file
        if chkp_file_path has existed, it will raise a error.
    CHKP_MODEL_LIST is a list:
        ['train.ckpt-3001', 'train.ckpt-1001', 'train.ckpt-2001', 'train.ckpt-3001']
    CHKP_FILE will be written: chkp_model_list[0] will as a index.
        model_checkpoint_path: "train.ckpt-3001"
        all_model_checkpoint_paths: "train.ckpt-1001"
        all_model_checkpoint_paths: "train.ckpt-2001"
        all_model_checkpoint_paths: "train.ckpt-3001"
    """
    gate.utils.filesystem.raise_path_exists(chkp_file_path)
    with open(chkp_file_path, 'w') as fp:
        fp.write('model_checkpoint_path: "'+chkp_model_list[0]+'"\n')
        for idx in range(1, len(chkp_model_list)):
            fp.write('all_model_checkpoint_paths: "'+chkp_model_list[idx]+'"\n')


def get_checkpoint_model_items(chkp_file_path):
    """ Extract checkpoint model items to list
    e.g.
        model_list[0]: model_checkpoint_path: "train.ckpt-3001"
        model_list[1]: all_model_checkpoint_paths: "train.ckpt-1001"
        model_list[2]: all_model_checkpoint_paths: "train.ckpt-2001"
        model_list[3]: all_model_checkpoint_paths: "train.ckpt-3001"
    """
    chkp_model_list = []
    with open(chkp_file_path) as fp:
        for line in fp:
            r = re.findall('\"(.*?)\"', line)
            if len(r):
                chkp_model_list.append(r[0])
    return chkp_model_list

import argparse
import torch
from MY_BERT import Constants
from transformers import BertTokenizer

def read_instances_from_file(data_file):
    ''' Convert file into word seq lists and vocab '''

    words_idx = []

    tag_lists = []
    # 创建tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    with open(data_file, 'r', encoding='utf-8') as f:
        #为了保持与bert模型训练时一致，前后都要加标签<cls>,<sep>
        word_idx = [Constants.CLS_id]  # 为防重复 只处理word  不处理label序列的前后
        tag_list = []  #
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                ###一个个转化
                indexed_tokens = tokenizer.convert_tokens_to_ids(word)

                word_idx.append(indexed_tokens)
                tag_list.append(tag)
            else:
                #加入<sep>
                word_idx.append(Constants.SEP_id)

                words_idx.append(word_idx)
                tag_lists.append(tag_list)

                word_idx = [Constants.CLS_id]
                tag_list = []



    print('[Info] Get {} instances from {}'.format(len(words_idx), data_file))
    max_sent_count=max([len(x) for x in words_idx])
    print('max_sent_length is  {}'.format(max_sent_count))

    return words_idx, tag_lists

def create_tag2idx():
    # 去除标签中所有重复元素，得到所有标签
    full_tags = ['B-CONT', 'E-CONT', 'M-CONT', 'B-EDU', 'E-EDU', 'M-EDU', 'B-LOC', 'E-LOC', 'M-LOC', 'B-NAME', 'E-NAME',
                 'M-NAME',
                 'S-NAME', 'B-ORG', 'E-ORG', 'M-ORG', 'S-ORG', 'B-PRO', 'E-PRO', 'M-PRO', 'B-RACE', 'E-RACE', 'M-RACE',
                 'S-RACE',
                 'B-TITLE', 'E-TITLE', 'M-TITLE']

    tag2idx = {
        Constants.OTHER_TAG: Constants.OTHER_TAG_id
    }

    for tag in full_tags:
        if tag not in tag2idx:
            tag2idx[tag] = len(tag2idx)

    classes=len(tag2idx)

    print('[Info] Get {} lables from file'.format(classes))

    return  tag2idx,classes

def create_tag_label(tag_lists,tag2idx,word_lists):

    labels_idx=[]

    for tags,words in zip(tag_lists,word_lists):
        sent_lable=[Constants.TAG_PAD_id]  #将首尾的CLS SEP 标签也算入pad
        for tag in tags :
            sent_lable.append(tag2idx[tag])

        sent_lable.append(Constants.TAG_PAD_id)
        labels_idx.append(sent_lable)

        if len(sent_lable) != len(words):
            print('警告:样本和标签数量不一致')

    return  labels_idx
#
# def convert_instance_to_idx_seq(word_insts, tag_insts):
#     ''' Mapping words to idx sequence. '''
#     full_tags = set(t for sent in tag_insts for t in sent) #去除标签中所有重复元素，得到所有标签
#     tag2idx={
#         Constants.zero_tag: 0
#     }
#     for tag in full_tags:
#         if tag not in tag2idx:
#             tag2idx[tag]=len(tag2idx)
#
#     word_ids=[]
#     label_ids=[]
#
#
#     # # Tokenize input
#     tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#     for sent,tags in zip(word_insts,tag_insts):
#
#         tokenized_text = tokenizer.tokenize(sent)
#         indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
#
#         word_ids.append(indexed_tokens)
#         #制作标签id
#         label_id=[tag2idx[tag] for tag in tags]
#         label_ids.append(label_id)
#
#         if len(label_id)!=len(indexed_tokens):
#             print('警告:样本和标签数量不一致')
#             print('text:',sent)
#             print('tokenized_text:',tokenized_text)
#             print('label:', len(label_id))
#
#
#
#     return  word_ids,label_ids,tag2idx


def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', default='../Data/train.char.bmes')
    parser.add_argument('-valid_src', default='../Data/dev.char.bmes')
    parser.add_argument('-test_src', default='../Data/test.char.bmes')
    parser.add_argument('-save_data', default='process_data.pth')
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=1)
    parser.add_argument('-keep_case', default=False)  # 区分大小写
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    #opt.max_token_seq_len = opt.max_word_seq_len + 1 # include the <s> and </s> here just <s>

    # Training set
    train_words_idx, train_labels= read_instances_from_file(opt.train_src)

    #create label dict
    tag2idx,lable_classes=create_tag2idx()

    #create label id
    train_labels_idx=create_tag_label(train_labels, tag2idx,train_words_idx)


    if len(train_words_idx) != len(train_labels_idx):
        print('[Warning] The training instance count is not equal.')
        print(len(train_words_idx),len(train_labels_idx))
        min_inst_count = min(len(train_words_idx), len(train_labels_idx))
        train_words_idx = train_words_idx[:min_inst_count]
        train_labels_idx = train_labels_idx[:min_inst_count]


    # #- Remove empty instances
    # train_words_idx, train_labels_idx = list(zip(*[
    #     (s, t) for s, t in zip(train_words_idx, train_labels_idx) if s and t]))
    # print(len(train_words_idx), len(train_labels_idx))

    # Validation set
    valid_words_idx, valid_tags = read_instances_from_file(
        opt.valid_src )

    #create label id
    valid_labels_idx=create_tag_label(valid_tags, tag2idx,valid_words_idx)

    if len(valid_words_idx) != len(valid_labels_idx):
        print('[Warning] The valid instance count is not equal.')
        min_inst_count = min(len(valid_words_idx), len(valid_tags))
        valid_words_idx = valid_words_idx[:min_inst_count]
        valid_labels_idx = valid_tags[:min_inst_count]

    # test set
    test_words_idx, test_tags = read_instances_from_file(opt.test_src )

    #create label id
    test_labels_idx=create_tag_label(test_tags, tag2idx,test_words_idx)

    if len(test_words_idx) != len(test_labels_idx):
        print('[Warning] The test instance count is not equal.')
        min_inst_count = min(len(test_words_idx), len(test_labels_idx))
        test_words_idx = test_words_idx[:min_inst_count]
        test_labels_idx = test_labels_idx[:min_inst_count]

    #
    # #- Remove empty instances
    # test_words, test_lables_id = list(zip(*[
    #     (s, t) for s, t in zip(test_words, test_lables_id) if s and t]))
    # word to index

    # train_src_insts,train_tags_id = convert_instance_to_idx_seq(train_words, train_tags,word2idx,tag2idx)
    # valid_src_insts,valid_tags_id = convert_instance_to_idx_seq(valid_words, valid_tags,word2idx,tag2idx)
    # test_src_insts,test_tags_id = convert_instance_to_idx_seq(test_words, test_tags,word2idx,tag2idx)

    data = {
        'settings': opt,
        'dict':{
            'tag2idx' :tag2idx
        },
        'lable_classes':lable_classes,
        'train': {
            'words': train_words_idx,
            'lables':train_labels_idx},
        'valid': {
            'words': valid_words_idx,
            'lables':valid_labels_idx},
        'test': {
            'words': test_words_idx,
            'lables':test_labels_idx}
    }

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    #main()
    #
    #   p
    path1='../../Data/test.char.bmes'
    path2='../process_data.pth'
    # data = torch.load(path1)
    # tag2id=data['dict']['tag2idx']
    words_idx,labels = read_instances_from_file(path1)
    #
    all_lables=set(tag for tags in labels for tag in tags )

    #print(tag2id)
    print(all_lables)


    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # # print(tokenizer.pad_token)
    # # print(tokenizer.pad_token_id)
    # sent='你'
    # sent=tokenizer.convert_tokens_to_ids(sent)
    # print(sent)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    #
    # #
    # # # Tokenize input
    #
    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # print(tokenized_text)
    # print(indexed_tokens)
    # ['[PAD]', '[CLS]', '[PAD]', '[SEP]']
    # # [0, 101, 0, 102]

    # #[Info] Get 3821 instances from ../Data/train.char.bmes
    # max_sent_length is  178
    # [Info] Get 28 lables from file
    # I1025 15:23:12.220628 140139995936512 tokenization_utils.py:373] loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt from cache at /home/wcy/.cache/torch/transformers/8a0c070123c1f794c42a29c6904beb7c1b8715741e235bee04aca2c7636fc83f.9b42061518a39ca00b8b52059fd2bede8daa613f8a8671500e518a8c29de8c00
    # [Info] Get 463 instances from ../Data/dev.char.bmes
    # max_sent_length is  178
    # I1025 15:23:37.949582 140139995936512 tokenization_utils.py:373] loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt from cache at /home/wcy/.cache/torch/transformers/8a0c070123c1f794c42a29c6904beb7c1b8715741e235bee04aca2c7636fc83f.9b42061518a39ca00b8b52059fd2bede8daa613f8a8671500e518a8c29de8c00
    # [Info] Get 477 instances from ../Data/test.char.bmes
    # max_sent_length is  167
    # [Info] Dumping the processed data to pickle file process_data.pth
    # [Info] Finish.


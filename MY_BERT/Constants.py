from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

PAD_id = tokenizer.pad_token_id
UNK_id= tokenizer.unk_token_id
CLS_id = tokenizer.cls_token_id
SEP_id = tokenizer.sep_token_id
TAG_PAD_id=100
OTHER_TAG_id=0


PAD_WORD = tokenizer.pad_token
UNK_WORD = tokenizer.unk_token
CLS_word = tokenizer.cls_token
SEP_word = tokenizer.sep_token

OTHER_TAG='O'
# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.data.ans_punct import prep_ans
import numpy as np
import en_vectors_web_lg, random, re, json


def shuffle_list(ans_list):
    random.shuffle(ans_list)


# ------------------------------
# ---- Initialization Utils ----
# ------------------------------

def img_feat_path_load(path_list):
    iid_to_path = {}

    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        iid_to_path[iid] = path

    return iid_to_path


def img_feat_tag_load(path_list):
    iid_to_feat = {}
    iid_to_tag = {}

    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        img = np.load(path)
        img_feat = img['x']#.transpose((1, 0))
        img_tag = img['tags']
        #########
        # print(ix," ",img_tag)
        #########


        iid_to_feat[iid] = img_feat
        iid_to_tag[iid] = img_tag
        print('\rPre-Loading: [{} | {}] '.format(ix, path_list.__len__()), end='          ')

    return iid_to_feat, iid_to_tag


def ques_load(ques_list):
    qid_to_ques = {}

    for ques in ques_list:
        qid = str(ques['question_id'])
        qid_to_ques[qid] = ques

    return qid_to_ques


def tokenize(stat_ques_list, tag_json_file, use_glove):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }

    spacy_tool = None
    pretrained_emb = []
    if use_glove:
        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)

    img_tag = json.load(open(tag_json_file, 'r'))
    for tag in img_tag:
        token_to_ix[tag] = len(token_to_ix)
        if use_glove:
            pretrained_emb.append(spacy_tool(tag).vector)

    for ques in stat_ques_list:
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb


# def ans_stat(stat_ans_list, ans_freq):
#     ans_to_ix = {}
#     ix_to_ans = {}
#     ans_freq_dict = {}
#
#     for ans in stat_ans_list:
#         ans_proc = prep_ans(ans['multiple_choice_answer'])
#         if ans_proc not in ans_freq_dict:
#             ans_freq_dict[ans_proc] = 1
#         else:
#             ans_freq_dict[ans_proc] += 1
#
#     ans_freq_filter = ans_freq_dict.copy()
#     for ans in ans_freq_dict:
#         if ans_freq_dict[ans] <= ans_freq:
#             ans_freq_filter.pop(ans)
#
#     for ans in ans_freq_filter:
#         ix_to_ans[ans_to_ix.__len__()] = ans
#         ans_to_ix[ans] = ans_to_ix.__len__()
#
#     return ans_to_ix, ix_to_ans


def ans_stat(json_file):
    ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))

    return ans_to_ix, ix_to_ans


# ------------------------------------
# ---- Real-Time Processing Utils ----
# ------------------------------------

def proc_img_feat(img_feat, img_object_pad_size):
    # print(img_object_pad_size)
    if img_feat.shape[0] > img_object_pad_size:
        img_feat = img_feat[:img_object_pad_size]

    img_feat = np.pad(
        img_feat,
        ((0, img_object_pad_size - img_feat.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return img_feat


def proc_img_tag(img_tag, token_to_ix, img_object_pad_size):
    # print("1 ", img_tag.shape)
    img_tag_ix = np.zeros(img_object_pad_size, np.int64)
    # print("2 ",img_tag_ix.shape)
    # input()

    for ix, tag in enumerate(img_tag):
        if tag in token_to_ix:
            img_tag_ix[ix] = token_to_ix[tag]
        else:
            img_tag_ix[ix] = token_to_ix['UNK']

        if ix + 1 == img_object_pad_size:
            break

    # print(img_tag_ix[:10])
    return img_tag_ix


def proc_ques(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)

    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques['question'].lower()
    ).replace('-', ' ').replace('/', ' ').split()

    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix


def get_score(occur):
    if occur == 0:
        return .0
    elif occur == 1:
        return .3
    elif occur == 2:
        return .6
    elif occur == 3:
        return .9
    else:
        return 1.


def proc_ans(ans, ans_to_ix):
    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
    ans_prob_dict = {}

    for ans_ in ans['answers']:
        ans_proc = prep_ans(ans_['answer'])
        if ans_proc not in ans_prob_dict:
            ans_prob_dict[ans_proc] = 1
        else:
            ans_prob_dict[ans_proc] += 1

    for ans_ in ans_prob_dict:
        if ans_ in ans_to_ix:
            ans_score[ans_to_ix[ans_]] = get_score(ans_prob_dict[ans_])

    return ans_score


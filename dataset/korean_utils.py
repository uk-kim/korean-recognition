import os
import re

BASE_CODE, INIT_CODE, MED_CODE = 44032, 588, 28


def list_to_string(s_list):
    string = ""
    for arg in s_list:
        string += arg
    return string


def save_korean_chars(path, init_list, med_list, final_list):
    with open(path, 'w') as f:
        i_string = list_to_string(init_list)
        m_string = list_to_string(med_list)
        f_string = list_to_string(final_list)

        f.write('%s\n' % i_string)
        f.write('%s\n' % m_string)
        f.write('%s\n' % f_string)


def load_korean_chars(path=None):
    if not path or not os.path.exists(path):
        i_list =['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
        m_list =['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ', 'ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
        f_list =[' ','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ', 'ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ',
                          'ㅅ', 'ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

    else:
        with open(path, 'r') as f:
            i_string = f.readline().strip()
            m_string = f.readline().strip()
            f_string = f.readline().strip()

            i_list = [c for c in i_string]
            m_list = [c for c in m_string]
            f_list = [""] + [c for c in f_string]

    return i_list, m_list, f_list


def decompose_korean_letter(ch, i2c_i, i2c_m, i2c_f, c2i_i, c2i_m, c2i_f):
    if len(ch) > 1:
        c1, c2, c3 = ch[0], ch[1], ch[2] if len(ch) == 3 else ''
        args = [c1, c2, c3]
        args_idx = [c2i_i[c1], c2i_m[c2], c2i_f[c3]]
    elif re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', ch) is not None:
        char_code = ord(ch) - BASE_CODE
        c1 = int(char_code / INIT_CODE)
        c2 = int((char_code - (INIT_CODE * c1)) / MED_CODE)
        c3 = int((char_code - (INIT_CODE * c1)) - (MED_CODE * c2))

        args = [i2c_i[c1], i2c_m[c2], i2c_f[c3]]
        args_idx = [c1, c2, c3]
    else:
        args = [ch]
        args_idx = [-1]

    return args, args_idx


def decompose_korean_letters(chs, i2c_i, i2c_m, i2c_f, c2i_i, c2i_m, c2i_f):
    args_list = []
    args_idx_list = []
    for ch in chs:
        args, args_idx = decompose_korean_letter(ch, i2c_i, i2c_m, i2c_f, c2i_i, c2i_m, c2i_f)
        args_list.append(args)
        args_idx_list.append(args_idx)

    return args_list, args_idx_list


def assemble_korean_letter(idxs):
    return BASE_CODE + idxs[0] * INIT_CODE + idxs[1] * MED_CODE + idxs[2]

import re
import numpy as np
import tensorflow as tf

SEQ_LENGTH=10

def mask_text(text):
    text = re.sub(r' and ', ' ', text)
    text = re.sub(r', ', ' ', text)
    combined_pattern = r"'[^']*'|(?<!<num)(?<!<var)\b(?:\d+\.?\d*|\.\d+)\b(?![^<>]*>)"
    mapping = {}
    counters = {'num': 1, 'var': 1}

    def replace_token(match):
        val = match.group(0)
        if val.startswith("'"):
            tag = f"<var{counters['var']}>"
            counters['var'] += 1
        else:
            tag = f"<num{counters['num']}>"
            counters['num'] += 1

        mapping[tag] = val
        return tag

    masked = re.sub(combined_pattern, replace_token, text)
    return masked, mapping


def unmask_text(predicted_text, mapping):
    # find tags like <var1> or <num2>
    tag_pattern = r'<(num|var)\d+>'

    def replace_match(match):
        tag = match.group(0).lower()
        return mapping.get(tag, tag)

    output_text = re.sub(tag_pattern, replace_match, predicted_text)
    return re.sub(r'\'', '', output_text)


def summarize_masked(text, vectorizer, model):
    vocab = vectorizer.get_vocabulary()
    m_text, mapping = mask_text(text)
    enc_input_vec = vectorizer([m_text])
    decoded_indices = [vocab.index("startseq")]

    for i in range(SEQ_LENGTH - 1):
        # Prepare decoder input: must match the shape (1, SEQ_LENGTH-1)
        dec_input_vec = np.zeros((1, SEQ_LENGTH - 1))
        for t, idx in enumerate(decoded_indices):
            if t < SEQ_LENGTH - 1:
                dec_input_vec[0, t] = idx

        # predict next token
        preds = model.predict([enc_input_vec, dec_input_vec], verbose=0)

        # get the index of the word with the highest probability
        # specifically for the current time step (len(decoded_indices) - 1)
        sampled_token_index = np.argmax(preds[0, len(decoded_indices) - 1, :])
        sampled_word = vocab[sampled_token_index]

        # Stop if end token is reached
        if sampled_word == "endseq" or sampled_word == "":
            break

        decoded_indices.append(sampled_token_index)

    # convert indices to words (skipping "startseq")
    predicted_words = [vocab[idx] for idx in decoded_indices[1:]]
    raw_prediction = " ".join(predicted_words)

    return unmask_text(raw_prediction, mapping)


def process_constrains(data_arr, unique_elements, exclude_keys = ['', 'main', 'cost']):
    result = []
    for element in unique_elements:
        if element in exclude_keys:
            continue
        filtered_element = data_arr[data_arr[:, -1] == element]
        lhs = f"{element}: "
        filter_notconst = filtered_element[filtered_element[:, 1] != 'constrain']
        values = filter_notconst[filter_notconst[:, 1] != 'cost']

        first_elem = True
        for value in values:
            if first_elem == False:
                lhs += " + "
            lhs += f"{value[0]} {value[1]}"
            first_elem = False

        constraints = filtered_element[filtered_element[:, 1] == 'constrain']
        rhs = ''
        # currently we are assuming we have only one constrain
        if len(constraints) > 1:
            print("TODO: error - update logic")
        if len(constraints) > 0:
            rhs = constraints[:, 0][0]

        result.append(f"{lhs} {rhs}")
    return result


def process_main(data_arr):
    result = []
    unique_types = np.unique(data_arr[:, -2])
    filtered_types = unique_types[unique_types != 'constrain']

    lhs = f"main: "
    first_elem = True
    for filtered_type in filtered_types:
        if first_elem == False:
            lhs += " + "
        lhs += f"{filtered_type}"
        first_elem = False

    filtered_element = data_arr[data_arr[:, -1] == 'main']
    rhs = filtered_element[:,0][0]
    result.append(f"{lhs} {rhs}")
    return result


def process_obj(data_arr):
    result = []
    unique_types = np.unique(data_arr[:, -2])
    filtered_types = unique_types[unique_types != 'constrain']
    lhs = f"obj: "
    first_elem = True
    for filtered_type in filtered_types:
        if first_elem == False:
            lhs += " + "

        # get cost numeric
        mask = np.isin(data_arr[:, -1], ['cost']) & (data_arr[:, 1] == filtered_type)
        filtered_element2 = data_arr[mask]
        lhs += f"{filtered_element2[0][0]}"
        lhs += f" {filtered_type}"
        first_elem = False

    result.append(f"{lhs} ")
    return result


def parse_unique_types(data_arr):
    result = ''
    unique_types = np.unique(data_arr[:, -2])
    filtered_types = unique_types[unique_types != 'constrain']

    for filtered_type in filtered_types:
        result += f"\n {filtered_type}"

    return result


def parse_dynamic_resources(data):
    lp_text = "\n\nMinimize\n"
    data1 = np.array(data)

    data_split = [item.split() for item in data1]
    max_cols = max(len(row) for row in data_split)
    padded_data = [row + [''] * (max_cols - len(row)) for row in data_split]
    data_arr = np.array(padded_data, dtype=object)

    unique_elements = np.unique(data_arr[:, -1])

    lp_text += process_obj(data_arr)[0]
    lp_text += "\n\nSubject To\n"
    lp_text += process_main(data_arr)[0]

    const_array = process_constrains(data_arr, unique_elements)
    for const in const_array:
        lp_text += f"\n{const}"

    lp_text += "\n\nBinary"
    lp_text += parse_unique_types(data_arr)
    lp_text += "\nEnd"
    return lp_text


def process_lp_vars(input_text, vectorizer, model):
    lp_vars = []
    segments = [s.strip() for s in re.split(r'(?<!\d)\.(?!\d)|[\n\r]+', input_text) if s.strip()]
    data_arr = np.array(segments)

    for data in data_arr:
        summrized_data = summarize_masked(data, vectorizer, model)
        summrized_sergments = [clean for s in summrized_data.split('. ') if (clean := s.strip())]
        for summrized_sergment in summrized_sergments:
            lp_vars.append(summrized_sergment)

    return lp_vars
